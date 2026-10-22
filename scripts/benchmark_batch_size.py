"""Measure the actual cached-feature CE + supervised audio-pair training step.

Weights/optimizer updates are disposable. No training checkpoint is overwritten.
The default scales the dedicated pair batch to preserve its ratio to main data.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from samsone_clm.common import write_json
from samsone_clm.contrast import (supervised_audio_partners, sample_audio_pair_batch,
                                 supervised_swap_loss)
from samsone_clm.model import AudioCLM
from samsone_clm.precompute import load_features
from samsone_clm.resources import require_resources
from samsone_clm.staged_train import parameter_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--feature-manifest', required=True)
    parser.add_argument('--sizes', type=int, nargs='+', default=[64, 128, 256, 512])
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--fixed-pair-batch', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 1 or min(args.sizes) < 1:
        parser.error('Sizes, warmup and steps must be positive')
    resources = require_resources()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    data, manifest = load_features(args.feature_manifest, dtype=torch.bfloat16)
    data = {key: tensor.cuda() for key, tensor in data.items()}
    train = torch.tensor([i for i, row in enumerate(manifest['rows']) if row['split'] == 'train'], device='cuda')
    data['partners'] = supervised_audio_partners(manifest['rows'], train.tolist(), 'reciprocal-answers').cuda()
    paired_rows = (data['partners'] >= 0).nonzero().flatten()
    if not len(paired_rows):
        raise ValueError('Benchmark requires verified audio pairs')
    results = []
    for batch_size in args.sizes:
        torch.manual_seed(42)
        model = AudioCLM.from_spec(checkpoint['spec']).cuda()
        model.load_state_dict(checkpoint['model_state'], strict=True)
        model.set_stage(4)
        model.train()
        optimizer = torch.optim.AdamW(parameter_groups(model, 1e-4, 5e-5), weight_decay=.01)
        pair_count = 32 if args.fixed_pair_batch else max(1, round(batch_size / 2))
        parameters = [p for p in model.parameters() if p.requires_grad]

        def update():
            main = train[torch.randint(len(train), (batch_size,), device='cuda')]
            pairs = sample_audio_pair_batch(data['partners'], paired_rows, pair_count)
            scores = model(data['correct'][main], data['actions'][data['indices'][main]])
            pair_scores = model(data['correct'][pairs], data['actions'][data['indices'][pairs]])
            loss = F.cross_entropy(scores, data['labels'][main])
            loss = loss + .5 * F.cross_entropy(pair_scores, data['labels'][pairs])
            loss = loss + supervised_swap_loss(model, data, pairs, pair_scores, .05)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            model.constrain_scale()
            return loss.detach(), norm.detach()

        try:
            for _ in range(args.warmup):
                update()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            for _ in range(args.steps):
                loss, norm = update()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise ValueError('Nonfinite benchmark loss or gradient')
            result = dict(batch_size=batch_size, pair_count=pair_count,
                          pair_endpoints=2*pair_count, steps=args.steps, elapsed_s=elapsed,
                          step_ms=1000*elapsed/args.steps,
                          main_samples_per_second=batch_size*args.steps/elapsed,
                          peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                          peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                          final_loss=float(loss), encoder_calls=0)
        except torch.OutOfMemoryError:
            result = dict(batch_size=batch_size, pair_count=pair_count, status='out_of_memory')
        results.append(result)
        print(json.dumps(result), flush=True)
        del model, optimizer, parameters
        gc.collect()
        torch.cuda.empty_cache()
        write_json(args.output, dict(device=torch.cuda.get_device_name(),
                                    precision='FP32 heads, BF16 feature storage, TF32 disabled',
                                    resources=resources, fixed_pair_batch=args.fixed_pair_batch,
                                    checkpoint=args.checkpoint, feature_manifest=args.feature_manifest,
                                    results=results, disposable_weights=True))


if __name__ == '__main__':
    main()
