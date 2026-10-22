"""Plan or launch additional full-dataset Stage 4 epochs in a separate run.

Stage 4 is already completed locally. This is an explicit additional experiment,
not an automatic resume of the completed common pipeline. No encoder is launched.
"""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from samsone_clm.common import PROJECT, ROOT, resolve_data_path


def training_command(args):
    config = json.loads((PROJECT / 'configs/migration.json').read_text())
    manifest = resolve_data_path(args.feature_manifest or ROOT / config['stage4_manifest'])
    output = Path(args.out or ROOT / 'runs/remote-stage4/stage4').expanduser().absolute()
    if output == ROOT / 'runs/v3/stage4' or output.is_relative_to(ROOT / 'models/samsone_clm_common'):
        raise ValueError('Use a separate output directory; the completed common model is immutable')
    command = [sys.executable, '-m', 'samsone_clm.train', '--stage', '4',
               '--feature-manifest', str(manifest), '--out', str(output), '--run-root', str(output.parent)]
    if args.resume:
        # Recover the exact saved optimizer/scheduler/objective rather than
        # substituting the wrapper defaults into a partially completed job.
        import torch
        saved = torch.load(output / 'last.pt', map_location='cpu', weights_only=True)
        settings = saved['args']
        command += ['--resume']
    else:
        settings = dict(epochs=args.epochs, batch_size=64, lr=args.lr, head_lr=args.head_lr,
                        loss='ce', scale_lr=.001, scheduler='onecycle', weight_decay=.01,
                        patience=args.patience, adapter_kind='linear', audio_swap_weight=1.,
                        audio_swap_margin=.05, audio_pair_batch_size=32, audio_pair_ce_weight=.5,
                        audio_pair_mode='reciprocal-answers', selection_metric='audio-grounded',
                        seed=42, steps=0, infonce_weight=1., learnable_scale=True)
        command += ['--continue-checkpoint', str(resolve_data_path(args.checkpoint or ROOT / config['common_best']))]
    for flag in ('epochs', 'batch_size', 'lr', 'head_lr', 'loss', 'scale_lr', 'scheduler',
                 'weight_decay', 'patience', 'adapter_kind', 'audio_swap_weight', 'audio_swap_margin',
                 'audio_pair_batch_size', 'audio_pair_ce_weight', 'audio_pair_mode', 'selection_metric',
                 'seed', 'steps', 'infonce_weight'):
        command += ['--' + flag.replace('_', '-'), str(settings[flag])]
    if settings['learnable_scale']:
        command += ['--learnable-scale']
    # One validation per complete train pass, as in v3 Stage 3/4.
    command += ['--evaluate-every', str(settings.get('evaluate_every') or
                   ((config['dataset_sizes']['train'] + settings['batch_size'] - 1) // settings['batch_size']))]
    command += ['--device', args.device]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', help='Raw Stage 4 checkpoint; defaults to exported common/best.pt')
    parser.add_argument('--feature-manifest')
    parser.add_argument('--out')
    parser.add_argument('--epochs', type=int, default=8, help='Additional epochs with a fresh optimizer/schedule')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--head-lr', type=float, default=5e-5)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', action='store_true', help='Exact resume of this new run from its last.pt')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.resume and args.checkpoint:
        parser.error('--resume cannot be combined with --checkpoint')
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    command = training_command(args)
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=PROJECT, check=True)


if __name__ == '__main__':
    main()
