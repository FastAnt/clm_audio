"""Compare full-clip and 10-second web-window predictions without reusing invalid crop labels."""
import argparse
from pathlib import Path
import time

import soundfile as sf
import torch

from samsone_clm.audio import decode_audio
from samsone_clm.common import ROOT,read_rows,digest,write_json
from samsone_clm.infer import Ranker
from samsone_clm.resources import require_resources
from samsone_clm.server import audio_window


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples-per-type',type=int,default=10)
    args=parser.parse_args()
    require_resources()
    torch.set_num_threads(2)
    rows=[r for r in read_rows(ROOT/'processed/v2/stage1_10k.jsonl') if r['split']=='validation']
    chosen=[]
    for kind in ('music','sound','speech','temporal'):
        chosen+=sorted([r for r in rows if r['question_type']==kind],key=lambda r:digest([601,r['sample_id']]))[:args.samples_per_type]
    ranker=Ranker(args.checkpoint)
    records=[]
    path=ROOT/'tmp/window_transfer.wav'
    for row in chosen:
        require_resources()
        waveform=decode_audio(row['audio_path'])
        position=len(waveform)/16000/2
        window,details=audio_window(waveform,position)
        sf.write(path,window,16000,subtype='FLOAT')
        full=ranker.rank(row['audio_path'],row['question'],row['choices'])
        cropped=ranker.rank(path,row['question'],row['choices'])
        records.append(dict(sample_id=row['sample_id'],question_type=row['question_type'],duration_s=len(waveform)/16000,
                            analyzed_window=details,full_ranking=full,window_ranking=cropped,
                            top1_changed=full[0]['choice']!=cropped[0]['choice']))
    path.unlink(missing_ok=True)
    report=dict(checkpoint=args.checkpoint,samples=len(records),prediction_change_rate=sum(r['top1_changed'] for r in records)/len(records),
                scope='Prediction stability only: original full-clip labels can become invalid after cropping or shifting temporal coordinates; window accuracy is not claimed',cases=records)
    write_json(args.output,report)
    print({k:v for k,v in report.items() if k!='cases'})


if __name__=='__main__':main()
