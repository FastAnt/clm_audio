import json
from datetime import datetime, timezone

from samsone_clm import monitor


def test_training_snapshot_preserves_actual_parameters_and_precompute_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor,"ROOT",tmp_path)
    runs=tmp_path/"runs"
    run=runs/"stage2"
    run.mkdir(parents=True)
    (tmp_path/"processed").mkdir()
    log=runs/"precompute_full.log"
    log.write_text("correct hidden 512/1000, elapsed=10s\ncorrect hidden 1000/1000\nsilence hidden 512/1000\n")
    (runs/"pipeline_status.json").write_text(json.dumps(dict(status="running",phase="precompute_full_strong_corpus",
         log=str(log),updated_at=datetime.now(timezone.utc).isoformat(),completed_stages=[1,2])))
    (tmp_path/"processed/full.report.json").write_text(json.dumps({"selected_samples":{"train":181592}}))
    metrics=dict(status="completed",stage=2,train_samples=50000,configuration={"lr":.0001,"head_lr":.0001,"batch_size":64,"epochs":8},
                 history=[dict(step=10,train_loss=1.2,validation={"top1":.62})],
                 trained_heads_changed={"state":True,"action":True},audio_dependence_gate_passed=False)
    (run/"metrics.json").write_text(json.dumps(metrics))
    snapshot=monitor.training_snapshot()
    assert snapshot["pipeline"]["phase"]=="precompute_full_strong_corpus"
    assert snapshot["precompute"]["correct"]==dict(done=1000,total=1000)
    assert snapshot["precompute_mode"]=="silence"
    stage=snapshot["stages"][1]
    assert stage["configuration"]==metrics["configuration"]
    assert stage["train_samples"]==50000
    assert stage["heads_changed"]==dict(state=True,action=True)
    assert stage["audio_dependence_gate_passed"] is False
    assert snapshot["stages"][3]["target_samples"]==181592
    assert snapshot["stages"][3]["status"]=="not_started"


def test_monitor_missing_artifacts_remain_unknown(tmp_path,monkeypatch):
    monkeypatch.setattr(monitor,"ROOT",tmp_path)
    snapshot=monitor.training_snapshot()
    assert snapshot["pipeline"]=={}
    assert all(s["status"]=="not_started" for s in snapshot["stages"])
    assert all(s["history"]==[] for s in snapshot["stages"])
    assert snapshot["stages"][3]["target_samples"] is None


def test_music_supervisor_takes_precedence_and_preserves_separate_stage4_run(tmp_path,monkeypatch):
    monkeypatch.setattr(monitor,'ROOT',tmp_path)
    music=tmp_path/'runs/music'
    music.mkdir(parents=True)
    previous=tmp_path/'runs/v3'
    previous.mkdir()
    (previous/'status.json').write_text(json.dumps({'phase':'completed','status':'completed'}))
    stage4=tmp_path/'runs/remote-stage4/stage4'
    stage4.mkdir(parents=True)
    (stage4/'metrics.json').write_text(json.dumps({'status':'completed','train_samples':181592}))
    (music/'status.json').write_text(json.dumps({'phase':'train-stage5','status':'running',
                    'run_root':str(music),'stage_directories':{'4':str(stage4)}}))
    stage5=music/'stage5'
    stage5.mkdir()
    (stage5/'progress.json').write_text(json.dumps({'step':25,'train_samples':92000,
                    'configuration':{'batch_size':256,'head_lr':.00005}}))
    snapshot=monitor.training_snapshot()
    assert snapshot['pipeline']['phase']=='train-stage5'
    assert len(snapshot['stages'])==6
    assert snapshot['stages'][3]['run_directory']==str(stage4)
    assert snapshot['stages'][4]['train_samples']==92000
    assert snapshot['stages'][4]['configuration']['batch_size']==256
    assert snapshot['stages'][5]['status']=='not_started'


def test_finalized_png_download_preserves_bundle_checksum(tmp_path,monkeypatch):
    from hashlib import sha256

    monkeypatch.setattr(monitor,'ROOT',tmp_path)
    run=tmp_path/'runs/music/stage6'
    run.mkdir(parents=True)
    image=run/'training_curves.png'
    payload=b'already finalized PNG contents'
    image.write_bytes(payload)
    digest=sha256(payload).hexdigest()
    (run/'checksums.json').write_text(json.dumps({'training_curves.png':digest}))
    (run/'metrics.json').write_text(json.dumps({'status':'completed','stage':6}))
    assert monitor.render_png(6,run)==image
    assert sha256(image.read_bytes()).hexdigest()==digest


def test_refinement_dashboard_uses_current_run_instead_of_completed_baseline(tmp_path,monkeypatch):
    monkeypatch.setattr(monitor,'ROOT',tmp_path)
    music=tmp_path/'runs/music'
    run=music/'stage6-full32'
    run.mkdir(parents=True)
    (music/'status.json').write_text(json.dumps({'phase':'completed','status':'completed'}))
    (music/'refinement_status.json').write_text(json.dumps({'phase':'train-stage6','status':'running',
                    'run_root':str(music),'stage_directories':{'6':str(run)}}))
    (run/'progress.json').write_text(json.dumps({'step':1490,'train_samples':190653}))
    snapshot=monitor.training_snapshot()
    assert snapshot['pipeline']['phase']=='train-stage6'
    assert snapshot['stages'][5]['run_directory']==str(run)
    assert snapshot['stages'][5]['progress']['step']==1490


def test_next_genre_precompute_keeps_completed_history_visible(tmp_path,monkeypatch):
    monkeypatch.setattr(monitor,'ROOT',tmp_path)
    base=tmp_path/'runs/music-genre'
    previous=base/'iteration-v1/stage6-genre-v1'
    pending=base/'iteration-v2/stage6-genre-v2'
    previous.mkdir(parents=True)
    (previous/'metrics.json').write_text(json.dumps(dict(stage=6,status='completed',
        train_samples=200936,history=[dict(step=785,train_loss=2.)])))
    (base/'status.json').write_text(json.dumps(dict(phase='precompute-large',status='running',
        run_root=str(base),stage_directories={'6':str(pending)})))
    snapshot=monitor.training_snapshot()
    stage=snapshot['stages'][5]
    assert stage['status']=='completed'
    assert stage['history'][0]['step']==785
    assert stage['train_samples']==200936
    assert stage['next_run_directory']==str(pending)
    assert snapshot['experiments'][0]['label']=='Stage 6 · FMA Medium · genre v1'
    assert monitor.resolve_experiment('music-genre/iteration-v1/stage6-genre-v1')==previous
    import pytest
    with pytest.raises(ValueError):
        monitor.resolve_experiment('../../etc')


def test_historical_genre_run_survives_new_optimizer_run(tmp_path,monkeypatch):
    monkeypatch.setattr(monitor,'ROOT',tmp_path)
    base=tmp_path/'runs/music-genre'
    previous=base/'iteration-v1/stage6-genre-v1'
    current=base/'iteration-v2/stage6-genre-v2'
    for p in [previous,current]:p.mkdir(parents=True)
    (previous/'metrics.json').write_text(json.dumps(dict(stage=6,status='completed',history=[dict(step=20)])))
    (current/'progress.json').write_text(json.dumps(dict(stage=6,status='running',step=1,train_samples=400000)))
    (base/'status.json').write_text(json.dumps(dict(phase='train-v2',status='running',
        run_root=str(base),stage_directories={'6':str(current)})))
    snapshot=monitor.training_snapshot()
    assert snapshot['stages'][5]['run_directory']==str(current)
    assert snapshot['stages'][5]['progress']['step']==1
    old=next(e for e in snapshot['experiments'] if e['run_directory']==str(previous))
    assert old['history'][0]['step']==20
