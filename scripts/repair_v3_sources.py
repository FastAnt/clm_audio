"""Repair audited TACOS source corruption from the verified original ZIP, keeping old artifacts."""
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path
from collections import defaultdict

import numpy as np
import soundfile as sf

from samsone_clm.common import ROOT,digest,read_rows,sha256,write_json
from samsone_clm.resources import require_resources


def main():
    require_resources()
    audit=json.loads((ROOT/'runs/v3/source_integrity.json').read_text())
    old=ROOT/'processed/v2'
    target=ROOT/'processed/v3'
    target.mkdir(parents=True,exist_ok=True)
    canonical=read_rows(old/'canonical.jsonl')
    changed={}
    repairs=[]
    if audit['mismatches']:
        if any(item['source_dataset']!='TACOS' for item in audit['mismatches']):
            raise RuntimeError('Non-TACOS corruption requires source-specific recovery; no guessed replacement')
        archive_path=ROOT/'raw/tacos_audio.zip'
        checksum=hashlib.md5()
        with archive_path.open('rb') as handle:
            for chunk in iter(lambda:handle.read(1024*1024),b''):checksum.update(chunk)
        if checksum.hexdigest()!='a1b11ed7a88d6b95109decb577c128fd':
            raise RuntimeError('Original TACOS ZIP failed published checksum')
        labels=read_rows(ROOT/'raw/AudioMCQ_TACOS.jsonl')
        official={digest([r['question'],r['choices'],r['answer']]) for r in labels
                  if r.get('audio-contribution',r.get('audio_contribution'))=='strong'}
        with zipfile.ZipFile(archive_path) as archive:
            for item in audit['mismatches']:
                name=item['source_audio_id']
                payload=archive.read(name)
                waveform,rate=sf.read(io.BytesIO(payload),dtype='float32',always_2d=True)
                waveform=waveform.mean(1)
                if not len(waveform) or not np.isfinite(waveform).all() or len(waveform)/rate>30.:
                    raise RuntimeError('Original source is invalid or exceeds 30s; refusing a label-changing crop')
                pcm='pcm:'+digest([rate,hashlib.sha256(waveform.tobytes()).hexdigest()])
                destination=ROOT/'processed/audio_verified/tacos'/f'{pcm[4:]}.wav'
                destination.parent.mkdir(parents=True,exist_ok=True)
                if not destination.exists():
                    temporary=destination.with_suffix('.wav.tmp')
                    sf.write(temporary,waveform,rate,subtype='FLOAT',format='WAV')
                    with temporary.open('rb') as handle:os.fsync(handle.fileno())
                    temporary.replace(destination)
                restored,restored_rate=sf.read(destination,dtype='float32')
                if restored_rate!=rate or not np.array_equal(restored,waveform):
                    raise RuntimeError('Restored PCM differs from original source')
                checksum_audio=sha256(destination)
                for row in item['rows']:
                    if digest([row['question'],row['choices'],row['choices'][row['answer_index']]]) not in official:
                        raise RuntimeError('Label no longer matches official AudioMCQ metadata')
                    fixed=dict(row,audio_path=str(destination),audio_sha256=checksum_audio,pcm_identity=pcm,
                               duration_s=len(waveform)/rate,sample_id=digest([pcm,row['question'],row['choices']]))
                    changed[row['sample_id']]=fixed
                repairs.append(dict(old_path=item['path'],new_path=str(destination),old_duration_s=item['rows'][0]['duration_s'],
                                    restored_duration_s=len(waveform)/rate,original_payload_sha256=hashlib.sha256(payload).hexdigest(),
                                    archive_md5=checksum.hexdigest(),old_pcm=item['rows'][0]['pcm_identity'],new_pcm=pcm,
                                    samples=len(item['rows']),labels_match_official_metadata=True,pcm_roundtrip_exact=True))
    repaired=[changed.get(row['sample_id'],row) for row in canonical]
    if len({r['sample_id'] for r in repaired})!=len(repaired):
        raise RuntimeError('Repair introduced duplicate IDs; requires explicit deduplication')
    pcm_splits=defaultdict(set)
    for row in repaired:pcm_splits[row['pcm_identity']].add(row['split'])
    if any(len(s)>1 for s in pcm_splits.values()):
        raise RuntimeError('Repaired PCM crosses splits; old checkpoints cannot be continued')
    def output(path,rows):
        text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
        if path.exists() and path.read_text()!=text:raise RuntimeError('Immutable repaired snapshot differs')
        if not path.exists():
            temporary=path.with_suffix('.jsonl.tmp');temporary.write_text(text);temporary.replace(path)
    output(target/'canonical.jsonl',repaired)
    snapshot_changes={}
    for name in ('pilot_1000','stage1_10k','stage2_50k','stage3_150k','stage4_all'):
        rows=read_rows(old/f'{name}.jsonl')
        new=[changed.get(r['sample_id'],r) for r in rows]
        heldout_changed=sum(r!=n for r,n in zip(rows,new) if r['split']!='train')
        modified=sum(r!=n for r,n in zip(rows,new))
        if heldout_changed:raise RuntimeError('Held-out data changed; must restart the comparison rather than silently continue')
        if name in ('pilot_1000','stage1_10k') and modified:
            raise RuntimeError('Stage1 data changed; must rerun overfit and Stage1 with new features')
        output(target/f'{name}.jsonl',new)
        snapshot_changes[name]=dict(changed_train_rows=modified,heldout_changed=heldout_changed)
    write_json(target/'repair_report.json',dict(canonical_sha256=sha256(target/'canonical.jsonl'),
                original_canonical_sha256=sha256(old/'canonical.jsonl'),repairs=repairs,snapshots=snapshot_changes,
                rows=len(repaired),heldout_unchanged=True,stage1_unchanged=True,pcm_cross_split_overlap=0))
    print(json.dumps(repairs,indent=2),flush=True)


if __name__=='__main__':main()
