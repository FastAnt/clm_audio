"""Measure full-excerpt versus padded central10s ranking on fixed validation QA.

This is a serving-distribution diagnostic, not checkpoint selection. Full-track
labels projected onto crops remain weak supervision; no test labels are read.
"""
import argparse
from collections import defaultdict
import json
import tempfile
from pathlib import Path

import soundfile as sf
import torch

from samsone_clm.audio import decode_audio
from samsone_clm.common import ROOT,digest,sha256,write_json
from samsone_clm.infer import Ranker
from samsone_clm.precompute import load_features
from samsone_clm.resources import require_resources
from samsone_clm.server import audio_window


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--feature-manifest',required=True)
    parser.add_argument('--samples-per-task',type=int,default=32)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/music/window_diagnostic.json')
    args=parser.parse_args()
    require_resources()
    torch.set_num_threads(4)
    data,manifest=load_features(args.feature_manifest,dtype=torch.bfloat16)
    ranker=Ranker(args.checkpoint)
    tasks={'instrument_family','instrument_source','vocals','mood_acoustic','tempo','meter','key_mode','key_tonic'}
    cohorts=defaultdict(list)
    for i,row in enumerate(manifest['rows']):
        if row['split']=='validation' and row.get('music_task') in tasks:
            cohorts[row['music_task']].append(i)
    results=[]
    with tempfile.TemporaryDirectory(dir=ROOT/'cache',prefix='window-diagnostic-') as scratch,torch.no_grad():
        for task,indices in sorted(cohorts.items()):
            indices=sorted(indices,key=lambda i:digest(['window-diagnostic-v1',manifest['rows'][i]['sample_id']]))[:args.samples_per_task]
            for i in indices:
                row=manifest['rows'][i]
                full=ranker.model(data['correct'][i:i+1].cuda(),data['actions'][data['indices'][i:i+1]].cuda(),
                                  data['candidate_mask'][i:i+1].cuda()).softmax(-1)[0,:len(row['choices'])].cpu()
                waveform=decode_audio(row['audio_path'])
                window,details=audio_window(waveform,len(waveform)/32000)
                path=Path(scratch)/'window.wav';sf.write(path,window,16000,subtype='FLOAT')
                ranking=ranker.rank(path,row['question'],row['choices'])
                probabilities={item['choice']:item['probability'] for item in ranking}
                cropped=torch.tensor([probabilities[c] for c in row['choices']])
                assert torch.isfinite(cropped).all() and abs(float(cropped.sum())-1)<1e-5
                results.append(dict(task=task,sample_id=row['sample_id'],audio=row['audio_path'],
                    source_duration_s=len(waveform)/16000,window=details,answer_index=row['answer_index'],
                    full_prediction=int(full.argmax()),window_prediction=int(cropped.argmax()),
                    full_correct_probability=float(full[row['answer_index']]),
                    window_correct_probability=float(cropped[row['answer_index']])))
            print(task,len(indices),'windows evaluated',flush=True)
    summary={}
    for task in sorted(cohorts):
        selected=[r for r in results if r['task']==task]
        summary[task]=dict(samples=len(selected),
            full_top1=sum(r['full_prediction']==r['answer_index'] for r in selected)/len(selected),
            window_top1=sum(r['window_prediction']==r['answer_index'] for r in selected)/len(selected),
            prediction_agreement=sum(r['full_prediction']==r['window_prediction'] for r in selected)/len(selected))
    write_json(args.output,dict(split='validation',checkpoint_sha256=sha256(args.checkpoint),
              samples=len(results),test_labels_used=False,used_for_checkpoint_selection=False,
              label_limit='Full-track labels on central crops remain weak; small diagnostic cohorts',
              summary=summary,results=results))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
