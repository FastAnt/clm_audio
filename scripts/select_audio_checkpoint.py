"""Select between retained best/last checkpoints using validation audio evidence.

This is checkpoint selection, not another training run. Source files are preserved.
"""
import argparse
import json
import shutil
from pathlib import Path

import torch

from samsone_clm.common import commit_file, write_json
from samsone_clm.metrics import evaluate_model
from samsone_clm.model import AudioCLM
from samsone_clm.precompute import load_features
from samsone_clm.resources import require_resources


def quality(evaluation):
    overall = evaluation['correct']['overall']
    gap = min(evaluation['audio_top1_gaps'].values(), default=float('-inf'))
    return (int(gap >= .05), overall['top1'], gap, -overall['loss'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--feature-manifest', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require_resources()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    source_report = json.loads((args.source/'metrics.json').read_text())
    if source_report.get('status') != 'completed':
        raise ValueError('Source training must be completed before checkpoint selection')
    if args.output.resolve() == args.source.resolve():
        raise ValueError('Selection must preserve the source training directory')
    data, manifest = load_features(args.feature_manifest, dtype=torch.bfloat16)
    data = {key: value.cuda() for key, value in data.items()}
    validation = torch.tensor([i for i, row in enumerate(manifest['rows'])
                               if row['split'] == 'validation'], device='cuda')
    candidates = {}
    for name in ('best.pt', 'last.pt'):
        checkpoint = torch.load(args.source/name, map_location='cpu', weights_only=True)
        if checkpoint['feature_provenance'] != manifest['provenance']:
            raise ValueError('Checkpoint/feature provenance mismatch')
        model = AudioCLM.from_spec(checkpoint['spec'])
        model.load_state_dict(checkpoint['model_state'], strict=True)
        model.cuda().eval()
        evaluation = evaluate_model(model, data, manifest['rows'], validation)
        candidates[name] = dict(step=checkpoint['step'], validation=evaluation,
                               score_scale=float(model.scale))
        del model, checkpoint
    selected = max(candidates, key=lambda name: quality(candidates[name]['validation']))
    args.output.mkdir(parents=True, exist_ok=True)
    temporary = args.output/'best.pt.tmp'
    shutil.copyfile(args.source/selected, temporary)
    commit_file(temporary, args.output/'best.pt')
    choice = dict(source_directory=str(args.source), selected_source=str(args.source/selected),
                  candidates=candidates, rule='Prefer both audio gaps >=5pp, then Top1, minimum gap, lower CE',
                  test_used=False, additional_training=False)
    write_json(args.output/'selection.json', choice)
    report = dict(source_report)
    # The source train evaluation belongs to source best, not necessarily selected last.
    report['evaluations'] = {'validation': candidates[selected]['validation']}
    report['best_step'] = candidates[selected]['step']
    report['score_scale'] = candidates[selected]['score_scale']
    report['audio_dependence_gate_passed'] = quality(candidates[selected]['validation'])[0] == 1
    report['checkpoint_selection'] = choice
    report['artifact_kind'] = 'selection among retained checkpoints; not a new training run'
    write_json(args.output/'metrics.json', report)
    print(json.dumps(dict(selected_checkpoint=str(args.output/'best.pt'),
                          validation=candidates[selected]['validation']['correct']['overall'],
                          audio_gaps=candidates[selected]['validation']['audio_top1_gaps'])), flush=True)


if __name__ == '__main__':
    main()
