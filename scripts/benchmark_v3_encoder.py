"""Choose a safe FP32 extraction batch and verify it preserves representations."""
import json
import argparse
import time
import torch
from samsone_clm.common import ROOT,read_rows,write_json,digest
from samsone_clm.encoder import FrozenSamsone
from samsone_clm.resources import require_resources

torch.set_num_threads(2)
parser=argparse.ArgumentParser()
parser.add_argument('--extended',action='store_true')
args=parser.parse_args()
require_resources()
rows=[r for r in read_rows(ROOT/'processed/v2/stage2_50k.jsonl') if r['split']=='train']
chosen=[]
for kind in ('music','sound','speech','temporal'):
    chosen.extend(sorted([r for r in rows if r['question_type']==kind],key=lambda r:digest([315,r['sample_id']]))[:64 if args.extended else 16])
encoder=FrozenSamsone()
measurements=[]
baseline=None
for batch in ((64,128,256) if args.extended else (16,32,64)):
    require_resources()
    torch.cuda.reset_peak_memory_stats()
    start=time.monotonic()
    values=torch.cat([encoder.state([r['audio_path'] for r in chosen[i:i+batch]], [r['question'] for r in chosen[i:i+batch]])
                      for i in range(0,len(chosen),batch)])
    elapsed=time.monotonic()-start
    if baseline is None: baseline=values
    error=float(((values-baseline).norm(dim=-1)/baseline.norm(dim=-1)).max())
    assert error<1e-4,error
    record=dict(batch_size=batch,seconds=elapsed,samples_per_second=len(chosen)/elapsed,
                peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,max_relative_l2=error)
    measurements.append(record)
    print(json.dumps(record),flush=True)
    write_json(ROOT/('runs/v3/encoder_benchmark_extended.json' if args.extended else 'runs/v3/encoder_benchmark.json'),measurements)
