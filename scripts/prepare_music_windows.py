"""Inherit source splits and labels for the demo's exact padded central10s view.

Half of TRAIN recordings receive this view; validation/test inherit the fixed
Stage5 cohorts. Notes remain intact. Jamendo crop labels are explicitly weak.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from samsone_clm.audio import decode_audio,write_audio_atomic
from samsone_clm.common import ROOT,digest,resolve_data_path,sha256,write_json
from samsone_clm.music_data import waveform_record,write_dataset
from samsone_clm.resources import require_resources
from samsone_clm.server import audio_window


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage5-manifest',required=True)
    args=parser.parse_args()
    require_resources()
    parent=json.loads(resolve_data_path(args.stage5_manifest).read_text())
    rows=[];clips={};silent=[]
    for row in parent['rows']:
        if not row.get('music_task'):
            continue
        if row['split']=='train' and int(digest(['music-window-view-v1',row['audio_identity']])[:8],16)%2:
            continue
        identity=row['audio_identity']
        if identity not in clips:
            waveform=decode_audio(row['audio_path'])
            window,details=audio_window(waveform,len(waveform)/32000)
            path=ROOT/'processed/music_windows/audio'/row['split']/(digest(identity)+'.wav')
            write_audio_atomic(path,window,16000)
            scope='whole_note_with_silence_padding' if row['source_dataset'].lower()=='nsynth' else 'full_track_consensus_projected_to_10s_weak'
            try:
                clips[identity]=waveform_record(path,identity,row['split_group'],row['split'],row['source_dataset'],scope,details)
            except ValueError as error:
                if 'Effectively silent source' not in str(error):raise
                clips[identity]=None;silent.append(identity)
        if clips[identity] is None:
            continue
        rows.append(dict(row,**clips[identity],parent_sample_id=row['sample_id'],
                         parent_audio_sha256=row['audio_sha256'],music_view='serving-window',
                         sample_id=digest([row['sample_id'],'padded-central-10s-v1']),
                         augmentation='exact server.audio_window at clip midpoint; source split inherited'))
    target=write_dataset(rows,'serving_windows.jsonl')
    report=json.loads(target.with_suffix('.report.json').read_text())
    report.update(parent_manifest_sha256=sha256(resolve_data_path(args.stage5_manifest)),
                  training_views='audio-identity hash selects half; remaining train recordings keep full excerpts',
                  audio_files=sum(v is not None for v in clips.values()),window_seconds=10,
                  excluded_silent_crop_identities=silent,
                  split_counts=dict(Counter(r['split'] for r in rows)))
    write_json(target.with_suffix('.report.json'),report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
