import io
import json
import tarfile

import numpy as np
import soundfile as sf

from samsone_clm import common, music_data


def test_nsynth_streaming_selection_keeps_official_instrument_splits(tmp_path,monkeypatch):
    monkeypatch.setattr(music_data,'ROOT',tmp_path)
    monkeypatch.setattr(common,'ROOT',tmp_path)
    monkeypatch.setattr(music_data,'require_resources',lambda:None)
    root=tmp_path/'raw/music/nsynth'
    root.mkdir(parents=True)
    (root/'provenance.json').write_text(json.dumps({'complete':True}))
    for instrument,split in enumerate(('train','valid','test')):
        metadata={}
        with tarfile.open(root/f'nsynth-{split}.jsonwav.tar.gz','w:gz') as tar:
            for note in range(2):
                name=f'guitar_acoustic_{instrument:03d}-{60+note:03d}-100'
                metadata[name]=dict(instrument=instrument,instrument_family_str='guitar',instrument_source_str='acoustic')
                wave=np.sin(2*np.pi*(220+instrument*50+note*10)*np.arange(64000)/16000).astype('float32')*.1
                buffer=io.BytesIO()
                sf.write(buffer,wave,16000,format='WAV',subtype='PCM_16')
                payload=buffer.getvalue()
                member=tarfile.TarInfo(f'nsynth-{split}/audio/{name}.wav')
                member.size=len(payload)
                tar.addfile(member,io.BytesIO(payload))
            payload=json.dumps(metadata).encode()
            member=tarfile.TarInfo(f'nsynth-{split}/examples.json')
            member.size=len(payload)
            tar.addfile(member,io.BytesIO(payload))
    rows=music_data.prepare_nsynth(notes_per_instrument=1)
    assert len(rows)==6
    assert all(row['split']=='train' for row in rows if row['official_source_split']=='train')
    assert all(row['split'] in ('validation','test') for row in rows if row['official_source_split']!='train')
    assert len({row['split_group'] for row in rows})==3
    assert all(row['duration_s']==4 and row['audio_num_samples']==64000 for row in rows)
    assert all(row['choices'][row['answer_index']] in ('guitar','acoustic') for row in rows)
    assert all(row['label_scope']=='entire isolated 4s note' for row in rows)
    assert all(common.resolve_data_path(row['audio_path']).exists() for row in rows)
    music_data.write_dataset(rows,'nsynth.jsonl')
    again=music_data.prepare_nsynth(notes_per_instrument=1)
    assert again==rows


def test_exact_pcm_duplicates_crossing_official_splits_are_excluded(tmp_path,monkeypatch):
    monkeypatch.setattr(music_data,'ROOT',tmp_path)
    rows=[]
    for index,split,pcm in [(0,'train','duplicate'),(1,'validation','duplicate'),(2,'train','unique')]:
        rows.append(dict(audio_identity=str(index),split_group=str(index),pcm_identity=pcm,split=split,
                         question='Which family?',choices=['guitar','flute'],answer_index=0,
                         sample_id=str(index),music_task='instrument_family'))
    target=music_data.write_dataset(rows,'fixture.jsonl')
    stored=[json.loads(line) for line in target.read_text().splitlines()]
    assert [row['sample_id'] for row in stored] == ['2']
    report=json.loads(target.with_suffix('.report.json').read_text())
    assert report['excluded_qa']==2
    assert report['cross_split_duplicate_pcm']==1
    assert report['split_leaks']==0
