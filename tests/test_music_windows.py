import importlib.util
import json
from pathlib import Path
import sys
import pytest
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from samsone_clm import common,music_data
from samsone_clm.metrics import music_selection_key
from samsone_clm.staged_train import ENCODER_FIELDS,validate_initialization


def test_short_note_window_keeps_label_and_instrument_split(tmp_path,monkeypatch):
    monkeypatch.setattr(common,'ROOT',tmp_path)
    monkeypatch.setattr(music_data,'ROOT',tmp_path)
    spec=importlib.util.spec_from_file_location('prepare_windows',Path(__file__).resolve().parents[1]/'scripts/prepare_music_windows.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'ROOT',tmp_path);monkeypatch.setattr(module,'require_resources',lambda:None)
    waveform=np.sin(np.arange(64000,dtype=np.float32)*.02)*.1
    source=tmp_path/'note.wav';sf.write(source,waveform,16000,subtype='FLOAT')
    base=music_data.waveform_record(source,'note:one','instrument:heldout','validation','NSynth','whole_note')
    parent=dict(base,sample_id='old-id',question='What instrument?',choices=['piano','flute'],answer_index=1,music_task='instrument_family')
    manifest=tmp_path/'parent.json';manifest.write_text(json.dumps({'rows':[parent]}))
    monkeypatch.setattr(sys,'argv',['prepare','--stage5-manifest',str(manifest)])
    module.main()
    row=common.read_rows(tmp_path/'processed/music/serving_windows.jsonl')[0]
    assert row['parent_sample_id']=='old-id' and row['sample_id']!='old-id'
    assert row['split_group']=='instrument:heldout' and row['split']=='validation'
    assert row['audio_identity']==parent['audio_identity'] and row['choices']==parent['choices'] and row['answer_index']==1
    assert row['label_scope']=='whole_note_with_silence_padding'
    audio,rate=sf.read(common.resolve_data_path(row['audio_path']),dtype='float32')
    assert rate==16000 and len(audio)==160000
    assert np.array_equal(audio[48000:112000],waveform)
    assert np.count_nonzero(audio[:48000])==np.count_nonzero(audio[112000:])==0


def test_window_selection_rejects_forgetting_and_audio_shortcuts():
    def report(full,window,gap=.1):
        def macro(score):return {'top1':score,'loss':.6,'family_top1':{'perception':score},'family_audio_top1_gaps':{'perception':{'random':gap,'silence':gap}}}
        return {'music_macro':macro((full+window)/2),'music_views':{
            'full':{'music_macro':macro(full)},'serving-window':{'music_macro':macro(window)}}}
    initial=report(.8,.5)
    accepted=report(.805,.6)
    assert music_selection_key(accepted,initial)[0]==1
    forgetting=report(.77,.9)  # Higher mixed accuracy cannot hide lost full-clip quality.
    assert music_selection_key(forgetting,initial)<music_selection_key(accepted,initial)
    shortcut=report(.81,.9)
    shortcut['music_views']['serving-window']['music_macro']['family_audio_top1_gaps']['perception']['silence']=0.
    assert music_selection_key(shortcut,initial)[0]==0
    assert music_selection_key(report(.8,.505),initial)[0]==0


def test_music_transfer_keeps_frozen_encoder_and_stage_guards():
    provenance={field:'unchanged' for field in ENCODER_FIELDS}
    manifest={'music_only':True,'provenance':provenance,'new_window_views':True}
    checkpoint={'stage':6,'feature_manifest':'old-full-excerpts','feature_provenance':dict(provenance)}
    with pytest.raises(ValueError):validate_initialization(checkpoint,manifest,6,continuation=True)
    validate_initialization(checkpoint,manifest,6,continuation=True,transfer_features=True)
    with pytest.raises(ValueError):validate_initialization(checkpoint,manifest,5,continuation=True,transfer_features=True)
    checkpoint['feature_provenance']['samsone_sha256']='different-encoder'
    with pytest.raises(ValueError,match='Frozen encoder'):validate_initialization(checkpoint,manifest,6,continuation=True,transfer_features=True)


def test_music_transfer_diagnostic_reaches_data_guard(monkeypatch):
    from samsone_clm import staged_train
    def data_guard():raise RuntimeError('launch guard passed')
    monkeypatch.setattr(staged_train,'require_resources',data_guard)
    args=SimpleNamespace(continue_checkpoint='same-stage-parent',resume=None,init_checkpoint=None,
                         stage=6,diagnostic=True,allow_feature_transfer=True)
    with pytest.raises(RuntimeError,match='launch guard passed'):staged_train.train_staged(args)
    args.allow_feature_transfer=False
    with pytest.raises(ValueError,match='Same-stage continuation'):staged_train.train_staged(args)
