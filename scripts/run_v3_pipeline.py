"""Resource-supervised AudioMCQ stages through full train, then common export.

Audio dependence is measured at every stage. The user explicitly requires
completion of Stage 4 before musical specialization starts.
"""
from __future__ import annotations
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

from samsone_clm.common import ROOT,PROJECT,digest,read_rows,write_json
from samsone_clm.encoder import FrozenSamsone
from samsone_clm.resources import require_resources,resource_status,probe_storage

RUN=ROOT/'runs/v3'
STATE=RUN/'status.json'


def main():
    probe_storage(RUN)
    previous_status=RUN/'status.json'
    if previous_status.exists():
        previous=json.loads(previous_status.read_text())
        pid=previous.get('supervisor_pid')
        if pid and pid!=os.getpid():
            try:
                command=Path(f'/proc/{pid}/cmdline').read_bytes().decode()
                if 'run_v3_pipeline.py' in command:
                    print(f'Existing v3 supervisor is active: PID {pid}',flush=True)
                    return
            except OSError:pass
    data_dir=ROOT/('processed/v3' if (ROOT/'processed/v3/repair_report.json').exists() else 'processed/v2')
    overfit=json.loads((RUN/'overfit-256-infonce/metrics.json').read_text())
    if not overfit.get('overfit_gate_passed') or not all(overfit['frozen_heads_unchanged'].values()):
        raise RuntimeError('Mechanical overfit gate has not passed; larger training is disabled')
    stage1=json.loads((RUN/'stage1/metrics.json').read_text())
    if stage1.get('status')!='completed' or any(stage1['trained_heads_changed'].values()):
        raise RuntimeError('A completed Stage1 with unchanged original heads is required')
    state=dict(status='running',run_root=str(RUN),completed_stages=[1],supervisor_pid=os.getpid(),
               completed_points=[1,2,3,4], encoder_recipe=FrozenSamsone.recipe,
               architecture='Frozen Samsone356M -> SAME adapter -> original CLM state/action heads',
               success_target_audio_gap=.05,completion_policy='Finish full available AudioMCQ Stage4 before music stages',
               model_name='samsone_clm_common')
    def heartbeat(phase,**extra):
        state.update(phase=phase,updated_at=datetime.now(timezone.utc).isoformat(),resources=resource_status(),**extra)
        write_json(STATE,state)
    def run(command,name,phase):
        require_resources()
        # Resume supervision without duplicating an already-running shard writer.
        existing=[]
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():continue
            try:
                argv=(entry/'cmdline').read_bytes().split(b'\0')
                argv=[v.decode() for v in argv if v]
                without_resume=[arg for arg in command if arg!='--resume']
                if argv==command or argv==without_resume:existing.append(int(entry.name))
            except (OSError,UnicodeError):pass
        if len(existing)>1:raise RuntimeError('Multiple identical workers found; refusing a duplicate writer')
        if existing:
            pid=existing[0]
            while Path(f'/proc/{pid}/cmdline').exists():
                try:
                    if not Path(f'/proc/{pid}/cmdline').read_bytes():break
                except OSError:break
                heartbeat(phase,active_pid=pid,log=str(RUN/name),adopted_existing_worker=True)
                require_resources(min_ram_gib=12.,min_disk_gib=12.)
                time.sleep(5)
            heartbeat(phase,active_pid=None)
            return
        with (RUN/name).open('a') as log:
            child=subprocess.Popen(command,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                while child.poll() is None:
                    heartbeat(phase,active_pid=child.pid,log=str(RUN/name),adopted_existing_worker=False)
                    require_resources(min_ram_gib=12.,min_disk_gib=12.)
                    time.sleep(5)
                if child.returncode:
                    raise RuntimeError(f'{phase} exited {child.returncode}; log={RUN/name}')
            except BaseException:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try: child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL)
                        child.wait()
                raise
            heartbeat(phase,active_pid=None)
    def matching_manifest(dataset):
        semantic=digest([{k:v for k,v in r.items() if k!='audio_sha256'} for r in read_rows(dataset)])
        for path in sorted((ROOT/'features').glob('spike-*/manifest.json')):
            manifest=json.loads(path.read_text())
            if (manifest['provenance']['recipe']==FrozenSamsone.recipe and
                manifest['provenance']['dataset_semantic_sha256']==semantic and
                (path.parent/'extraction_audit.json').exists()):
                return str(path)
    def features(snapshot,batch):
        dataset=data_dir/f'{snapshot}_audio_contrast_reciprocal.jsonl'
        if not dataset.exists():
            run([sys.executable,'scripts/prepare_audio_contrast.py','--pair-mode','reciprocal-answers',
                 '--base-snapshot',snapshot,'--data-dir',str(data_dir)],f'prepare_{snapshot}.log',f'preparing_{snapshot}')
        path=matching_manifest(dataset)
        if not path:
            run([sys.executable,'scripts/precompute.py','--dataset',str(dataset),'--batch-size',str(batch),
                 '--shard-size','512','--ablation-splits','validation','test'],
                f'precompute_{snapshot}.log',f'precompute_{snapshot}')
            path=matching_manifest(dataset)
        if not path: raise RuntimeError('No complete verified feature manifest')
        return path
    def staged(stage,manifest,loss,diagnostic=False,*,name=None,init=None):
        directory=RUN/(name or (f'diagnostic-stage2-{loss}' if diagnostic else f'stage{stage}'))
        report=directory/'metrics.json'
        if report.exists() and json.loads(report.read_text()).get('status')=='completed':
            return json.loads(report.read_text())
        command=[sys.executable,'-m','samsone_clm.train','--stage',str(stage),'--epochs','12',
                 '--loss',loss,'--learnable-scale','--scale-lr','0.001','--scheduler','onecycle',
                 '--weight-decay','0.01','--patience','4','--adapter-kind','linear','--lr','0.0003',
                 '--head-lr','0.0001','--audio-swap-weight','1','--audio-pair-batch-size','32',
                 '--audio-pair-ce-weight','0.5','--audio-pair-mode','reciprocal-answers',
                 '--feature-manifest',manifest,'--run-root',str(RUN),'--out',str(directory)]
        if stage>=3:
            main_count=sum(row['split']=='train' and row.get('training_role')!='contrast_only'
                           for row in json.loads(Path(manifest).read_text())['rows'])
            command+=['--selection-metric','audio-grounded','--evaluate-every',str((main_count+63)//64)]
        if diagnostic: command+=['--diagnostic','--samples','10000','--init-checkpoint',str(RUN/'stage1/best.pt')]
        elif init:command+=['--init-checkpoint',str(init)]
        if (directory/'last.pt').exists(): command+=['--resume']
        run(command,directory.name+'.log','training_'+directory.name)
        return json.loads(report.read_text())
    def finish(checkpoint,manifest,reason):
        state['final_raw_checkpoint']=str(checkpoint)
        run([sys.executable,'scripts/calibrate_checkpoint.py','--checkpoint',str(checkpoint),
             '--feature-manifest',manifest],'calibration.log','calibrating_probabilities_on_validation')
        calibrated=checkpoint.parent/'calibrated.pt'
        state['final_checkpoint']=str(calibrated)
        run([sys.executable,'-m','samsone_clm.evaluate','--checkpoint',str(checkpoint),
             '--feature-manifest',manifest,'--split','test','--output',str(RUN/'raw_test.json')],
            'final_raw_test.log','final_test_evaluation')
        run([sys.executable,'-m','samsone_clm.evaluate','--checkpoint',str(calibrated),
             '--feature-manifest',manifest,'--split','test','--output',str(RUN/'calibrated_test.json')],
            'final_test.log','final_test_evaluation')
        run([sys.executable,'scripts/verify_audio_contrast_inference.py','--checkpoint',str(calibrated),
             '--output',str(RUN/'live_inference.json')],'live_inference.log','checking_real_audio_inference')
        run([sys.executable,'scripts/evaluate_audio_transfer.py','--checkpoint',str(calibrated),
             '--feature-manifest',manifest,'--output',str(RUN/'paired_audio_evaluation.json')],
            'paired_audio_evaluation.log','checking_audio_transfer')
        run([sys.executable,'scripts/check_window_transfer.py','--checkpoint',str(calibrated),
             '--output',str(RUN/'window_transfer.json')],'window_transfer.log','checking_10s_window_transfer')
        temporary=RUN/'demo-next.pt'
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(calibrated)
        temporary.replace(RUN/'demo.pt')
        run([sys.executable,'scripts/verify_web.py','--dataset',str(ROOT/'processed/v2/stage1_10k.jsonl')],
            'final_web_verification.log','checking_final_web_demo')
        run([sys.executable,'scripts/report_v3.py'],'report.log','writing_report')
        run([sys.executable,'scripts/export_common.py','--run-root',str(RUN),
             '--checkpoint',str(checkpoint),'--feature-manifest',manifest],
            'export_common.log','saving_samsone_clm_common')
        heartbeat('completed',status='completed',active_pid=None,supervisor_pid=None,
                  conclusion=reason,completed_points=[1,2,3,4,5,6,7,8,9],
                  exported_model=str(ROOT/'models/samsone_clm_common'),
                  exported_dataset=str(ROOT/'processed/samsone_clm_common'))
    try:
        manifest10=str(ROOT/'features/spike-5ec3f92322401b72/manifest.json')
        ce=staged(2,manifest10,'ce',True)
        contrast=staged(2,manifest10,'infonce',True)
        official=staged(2,manifest10,'clm-infonce',True)
        pilots={'ce':ce,'infonce':contrast,'clm-infonce':official}
        def quality(report):
            ev=report['evaluations']['validation']
            gap=min(ev['audio_top1_gaps'].values())
            return (int(gap>=.05),ev['correct']['overall']['top1'],gap,-ev['correct']['overall']['loss'])
        chosen=max(pilots,key=lambda key:quality(pilots[key]))
        # A text-only winner must not erase the strongest audio-dependent pilot.
        if min(pilots[chosen]['evaluations']['validation']['audio_top1_gaps'].values())<=0:
            chosen=max(pilots,key=lambda key:min(pilots[key]['evaluations']['validation']['audio_top1_gaps'].values()))
        write_json(RUN/'recipe_comparison.json',dict(selected_loss=chosen,selection_rule='Prefer passed 5pp audio gate; then Top1, minimum audio gap, CE; require positive gaps',
                   runs={k:dict(validation=v['evaluations']['validation'],score_scale=v['score_scale'],best_step=v['best_step']) for k,v in pilots.items()}))
        heartbeat('recipe_comparison_completed',selected_loss=chosen,completed_points=[1,2,3,4,5,6])
        if not (RUN/'encoder_benchmark.json').exists():
            run([sys.executable,'scripts/benchmark_v3_encoder.py'],'encoder_benchmark.log','benchmarking_safe_encoder_batch')
        measurements=json.loads((RUN/'encoder_benchmark.json').read_text())
        batch=max(measurements,key=lambda item:item['samples_per_second'])['batch_size']
        heartbeat('batch_selected',precompute_batch_size=batch)
        manifest=features('stage2_50k',batch)
        large=staged(2,manifest,chosen)
        large_directory=RUN/'stage2'
        if not large['audio_dependence_gate_passed']:
            alternative='clm-infonce' if chosen=='ce' else 'ce'
            candidate=staged(2,manifest,alternative,name='stage2-'+alternative)
            write_json(RUN/'scale_recipe_comparison.json',dict(primary_loss=chosen,alternative_loss=alternative,
                       primary_validation=large['evaluations']['validation'],alternative_validation=candidate['evaluations']['validation'],
                       selection_rule='Top1, minimum audio gap, then CE on validation; no test selection'))
            if quality(candidate)>quality(large) and min(candidate['evaluations']['validation']['audio_top1_gaps'].values())>0:
                large=candidate;chosen=alternative;large_directory=RUN/('stage2-'+alternative)
        # The same predeclared audio-first rule also applies to retained last.pt.
        # Keep source best/last and its original metrics untouched.
        retained=[]
        sources=[(RUN/'stage2',state['selected_loss'])]
        if (RUN/'stage2-clm-infonce/metrics.json').exists():sources.append((RUN/'stage2-clm-infonce','clm-infonce'))
        if (RUN/'stage2-ce/metrics.json').exists():sources.append((RUN/'stage2-ce','ce'))
        for source,source_loss in sources:
            selected_directory=RUN/(source.name+'-audio-grounded')
            if not (selected_directory/'metrics.json').exists():
                run([sys.executable,'scripts/select_audio_checkpoint.py','--source',str(source),
                     '--feature-manifest',manifest,'--output',str(selected_directory)],
                    selected_directory.name+'.log','selecting_retained_audio_checkpoint')
            retained.append((json.loads((selected_directory/'metrics.json').read_text()),source_loss,selected_directory))
        large,chosen,large_directory=max(retained+[(large,chosen,large_directory)],key=lambda item:quality(item[0]))
        state['selected_loss']=chosen
        state['stage_directories']={'2':str(large_directory)}
        state['completed_stages']=[1,2]
        previous=pilots[chosen]['evaluations']['validation']
        current=large['evaluations']['validation']
        # Predeclared evidence gate, checked only on validation, never on test.
        improving=(current['correct']['overall']['top1']>=previous['correct']['overall']['top1']+.005
                   and min(current['audio_top1_gaps'].values())>=max(.02,min(previous['audio_top1_gaps'].values())))
        write_json(RUN/'scale_decision.json',dict(stage2_validation=current,pilot_validation=previous,
                   continue_to_150k=True,prior_evidence_gate_passed=improving,
                   rule='User requires full available AudioMCQ completion; prior evidence gate is recorded as an outcome',
                   five_point_success_target=large['audio_dependence_gate_passed']))
        heartbeat('stage2_checkpoint_selected',completed_points=[1,2,3,4,5,6,7])
        previous=current
        last_accepted_checkpoint=large_directory/'best.pt'
        last_accepted_manifest=manifest
        for stage,snapshot in [(3,'stage3_150k'),(4,'stage4_all')]:
            manifest=features(snapshot,batch)
            report=staged(stage,manifest,chosen,init=last_accepted_checkpoint)
            state['completed_stages'].append(stage)
            current=report['evaluations']['validation']
            write_json(RUN/f'stage{stage}_scale_decision.json',dict(
                validation=current,preceding_validation=previous,
                accuracy_regressed=current['correct']['overall']['top1']<previous['correct']['overall']['top1'],
                audio_gate_passed=report['audio_dependence_gate_passed'],
                continue_to_full_audiomcq=True,policy='Complete full AudioMCQ per user instruction; report regression without stopping dataset completion'))
            previous=current
            last_accepted_checkpoint=RUN/f'stage{stage}/best.pt'
            last_accepted_manifest=manifest
        finish(RUN/'stage4/best.pt',manifest,'Full available AudioMCQ completed and exported as samsone_clm_common; music stages are separate and have not started')
    except Exception as error:
        heartbeat('failed',status='failed',active_pid=None,supervisor_pid=None,error=str(error))
        raise


if __name__=='__main__': main()
