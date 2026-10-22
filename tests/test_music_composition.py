import importlib.util
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import save_file

from samsone_clm import common
from samsone_clm.precompute import load_features
from samsone_clm.staged_train import ENCODER_FIELDS


def test_replay_uses_only_verified_cached_features_without_source_audio(tmp_path,monkeypatch):
    monkeypatch.setattr(common,'ROOT',tmp_path)
    spec=importlib.util.spec_from_file_location('compose_music',Path(__file__).resolve().parents[1]/'scripts/compose_music_features.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module,'require_resources',lambda:None)
    paths=[]
    for component in ('new','common'):
        directory=tmp_path/component
        directory.mkdir()
        rows=[]
        for split in ('train','validation','test'):
            for label in range(2):
                identity=f'{component}:{split}:{label}'
                rows.append(dict(sample_id=identity,audio_identity=identity,pcm_identity=identity,split_group=identity,
                                 audio_path='/missing/source.wav',split=split,answer_index=label,
                                 choices=['yes','no'],question_type='music',
                                 **({'music_task':'vocals'} if component=='new' else {})))
        provenance={key:'fixture' for key in ENCODER_FIELDS}
        provenance['hidden_width']=3
        for mode in ('correct','random','silence','actions'):
            hidden=torch.ones(2 if mode=='actions' else 6,3,dtype=torch.bfloat16)
            save_file({'hidden':hidden},str(directory/f'{mode}-000000.safetensors'))
        common.write_json(directory/'manifest.json',dict(directory=str(directory),rows=rows,choices=['yes','no'],
              candidate_indices=[[0,1]]*6,provenance=provenance,
              ablation_available=[False,False,True,True,True,True],
              shard_sha256={p.name:common.sha256(p) for p in directory.glob('*.safetensors')}))
        paths.append(str(directory/'manifest.json'))
    monkeypatch.setattr(sys,'argv',['compose','--stage','5','--new-manifests',paths[0],'--common-manifest',paths[1]])
    module.main()
    pointer=json.loads((tmp_path/'features/music_stage5_manifest.json').read_text())
    data,manifest=load_features(pointer['path'],dtype=torch.bfloat16)
    assert manifest['encoder_calls']==0
    assert len(manifest['components'])==2
    assert data['correct'].shape==(11,3)  # all six new rows + one train/four held-out replay rows
    assert manifest['directory'].startswith(str(common.ORIGINAL_DATA_ROOT))
    assert manifest['music_only']


def test_window_replay_replaces_train_views_and_keeps_full_heldout_cohorts(tmp_path,monkeypatch):
    monkeypatch.setattr(common,'ROOT',tmp_path)
    spec=importlib.util.spec_from_file_location('compose_windows',Path(__file__).resolve().parents[1]/'scripts/compose_music_features.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'ROOT',tmp_path);monkeypatch.setattr(module,'require_resources',lambda:None)
    def store(name,rows,value):
        directory=tmp_path/name;directory.mkdir()
        for mode in ('correct','random','silence','actions'):
            hidden=torch.full((2 if mode=='actions' else len(rows),3),1 if mode=='actions' else value,dtype=torch.bfloat16)
            save_file({'hidden':hidden},str(directory/f'{mode}-000000.safetensors'))
        provenance={key:'fixture' for key in ENCODER_FIELDS};provenance['hidden_width']=3
        path=directory/'manifest.json'
        common.write_json(path,dict(directory=str(directory),rows=rows,choices=['yes','no'],
            candidate_indices=[[0,1]]*len(rows),provenance=provenance,ablation_available=[True]*len(rows),
            shard_sha256={p.name:common.sha256(p) for p in directory.glob('*.safetensors')}))
        return str(path)
    def rows(name):
        return [dict(sample_id=f'{name}:{split}:{label}',audio_identity=f'{name}:{split}:{label}',
            pcm_identity=f'{name}:{split}:{label}',split_group=f'{name}:{split}:{label}',
            audio_path='/absent.wav',split=split,answer_index=label,choices=['yes','no'],
            question='Voice?',question_type='music',music_task='vocals')
            for split in ('train','validation','test') for label in range(2)]
    parents=rows('replay')
    windows=[dict(parents[i],sample_id='window:'+parents[i]['sample_id'],
                 parent_sample_id=parents[i]['sample_id'],pcm_identity='window-pcm:'+parents[i]['pcm_identity'],
                 music_view='serving-window') for i in (0,2,4)]
    new=store('new',rows('new'),2);replay=store('replay',parents,3)
    prior=store('common',rows('common'),4);window=store('window',windows,7)
    pointer=tmp_path/'features/separate_window_pointer.json'
    original=tmp_path/'features/music_stage6_manifest.json';original.parent.mkdir();original.write_text('unchanged')
    monkeypatch.setattr(sys,'argv',['compose','--stage','6','--new-manifests',new,
        '--common-manifest',prior,'--stage5-manifest',replay,'--replay-window-manifest',window,'--pointer',str(pointer)])
    module.main()
    data,manifest=load_features(json.loads(pointer.read_text())['path'],dtype=torch.bfloat16)
    ids={r['sample_id']:i for i,r in enumerate(manifest['rows'])}
    assert 'replay:train:0' not in ids and 'window:replay:train:0' in ids
    assert all(r['sample_id'] in ids for r in parents if r['split']!='train')
    assert torch.equal(data['correct'][ids['window:replay:train:0']],torch.full((3,),7,dtype=torch.bfloat16))
    assert sum(r['split']=='train' for r in manifest['rows'])==5
    assert original.read_text()=='unchanged'
