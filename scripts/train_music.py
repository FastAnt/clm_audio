"""Stage 5/6 recipe. Overfit new music before large training; immutable common parent."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

from samsone_clm.common import ROOT, digest, resolve_data_path, write_json
from samsone_clm.precompute import load_features
from samsone_clm.resources import require_resources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',type=int,choices=[5,6],required=True)
    parser.add_argument('--feature-manifest')
    parser.add_argument('--parent-checkpoint')
    parser.add_argument('--run-root',type=Path,default=ROOT/'runs/music')
    parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--epochs',type=int,default=32)
    parser.add_argument('--lr',type=float,default=.0001)
    parser.add_argument('--head-lr',type=float,default=.00005)
    parser.add_argument('--scale-lr',type=float,default=.001)
    parser.add_argument('--patience',type=int,default=5)
    parser.add_argument('--run-name',help='Separate experimental run under run-root; does not overwrite stage5/6')
    parser.add_argument('--selection-metric',choices=['music-grounded','music-window-grounded','genre-grounded'],default='music-grounded')
    parser.add_argument('--balance-genre-ce',action='store_true')
    parser.add_argument('--diagnostic',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    require_resources()
    feature_path = args.feature_manifest or json.loads((ROOT/f'features/music_stage{args.stage}_manifest.json').read_text())['path']
    manifest = json.loads(resolve_data_path(feature_path).read_text())
    parent = resolve_data_path(args.parent_checkpoint or
              (ROOT/'models/samsone_clm_common/best.pt' if args.stage==5 else args.run_root/'stage5/best.pt'))
    train_count = sum(r['split']=='train' and r.get('training_role')!='contrast_only' for r in manifest['rows'])
    if args.run_name and (Path(args.run_name).name!=args.run_name or args.run_name in ('.','..')):
        raise ValueError('Run name must be a single directory name')
    run = args.run_root/(args.run_name or (f'diagnostic-stage{args.stage}-512' if args.diagnostic else f'stage{args.stage}'))
    gate_path = args.run_root/f'overfit_stage{args.stage}_gate.json'
    if not args.diagnostic:
        gate = json.loads(gate_path.read_text())
        if not gate['passed'] or gate['feature_manifest'] != digest(manifest):
            raise ValueError('Pass the new-music overfit gate on the exact feature manifest first')
    batch = 64 if args.diagnostic else args.batch_size
    epochs = 256 if args.diagnostic else args.epochs
    command = [sys.executable,'-m','samsone_clm.train','--stage',str(args.stage),
               '--feature-manifest',str(feature_path),'--out',str(run),
               '--run-root',str(args.run_root),'--batch-size',str(batch),'--epochs',str(epochs),
               '--lr','.0003' if args.diagnostic else str(args.lr),
               '--head-lr','.0001' if args.diagnostic else str(args.head_lr),'--learnable-scale','--scale-lr',str(args.scale_lr),
               '--adapter-kind','linear','--weight-decay','.01','--loss','ce',
               '--scheduler','onecycle',
               '--patience','0' if args.diagnostic else str(args.patience),
               '--selection-metric','accuracy' if args.diagnostic else args.selection_metric,
               '--evaluate-every',str(100 if args.diagnostic else math.ceil(train_count/batch)),
               '--audio-swap-weight','0' if args.diagnostic else '1',
               '--audio-pair-batch-size','0' if args.diagnostic else str(max(1,batch//2)),
               '--audio-pair-ce-weight','.5','--audio-swap-margin','.05',
               '--audio-pair-mode','reciprocal-diverse']
    if args.balance_genre_ce and not args.diagnostic:
        command += ['--balance-genre-ce']
    if args.diagnostic:
        command += ['--diagnostic','--samples','512','--steps','2048']
    report = json.loads((run/'metrics.json').read_text()) if (run/'metrics.json').exists() else {}
    if args.resume and report.get('status') == 'completed':
        print(f'Already completed: {run}',flush=True)
    else:
        if (run/'last.pt').exists():
            if not args.resume:
                raise ValueError('Existing run requires --resume')
            command += ['--resume']
        else:
            parent_stage=int(torch.load(parent,map_location='cpu',weights_only=True,mmap=True)['stage'])
            command += (['--continue-checkpoint',str(parent),'--allow-feature-transfer'] if parent_stage==args.stage else
                        ['--init-checkpoint',str(parent)])
        print('Launching:', ' '.join(command),flush=True)
        subprocess.run(command,check=True)
        report = json.loads((run/'metrics.json').read_text())
    if args.diagnostic:
        # Check last.pt too: validation selection can prefer an earlier, less
        # memorized model. This is a mechanical TRAIN loss check, not val/test selection.
        from samsone_clm.model import AudioCLM
        from samsone_clm.metrics import evaluate_model
        data, manifest = load_features(feature_path,dtype=torch.bfloat16)
        data = {key:value.cuda() for key,value in data.items()}
        checkpoint = torch.load(run/'last.pt',map_location='cpu',weights_only=True)
        ids = {r['sample_id']:i for i,r in enumerate(manifest['rows'])}
        indices = torch.tensor([ids[s] for s in checkpoint['main_train_sample_ids']],device='cuda')
        model = AudioCLM.from_spec(checkpoint['spec']).cuda().eval()
        model.load_state_dict(checkpoint['model_state'],strict=True)
        evaluation = evaluate_model(model,data,manifest['rows'],indices,batch)
        final = evaluation['correct']['overall']
        initial_ce = report['history'][0]['train_loss']
        passed = final['top1'] >= .95 and final['loss'] < min(.2,initial_ce*.5)
        write_json(gate_path,dict(passed=passed,feature_manifest=digest(manifest),
                   diagnostic_checkpoint_sha256=__import__('samsone_clm.common',fromlist=['sha256']).sha256(run/'last.pt'),
                   final_train=final,first_interval_ce=initial_ce,
                   both_heads_train=True,encoder_calls=0,feature_manifest_path=str(feature_path)))
        print('OVERFIT GATE:',passed,final,flush=True)
        if not passed:
            raise RuntimeError('New-music overfit did not pass; larger training is stopped')


if __name__ == '__main__':
    main()
