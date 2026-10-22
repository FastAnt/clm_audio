"""Local supervisor for REMOTE music stages, including verified artifact return.

All model/precompute/training commands run through SSH on the specified host.
Local work is limited to orchestration, checksum verification and backups.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from samsone_clm.common import ROOT, write_json


class Pipeline:
    def __init__(self,args):
        self.args=args
        self.ssh=['ssh','-p',str(args.port),'-o','BatchMode=yes','-o','ConnectTimeout=20',
                  '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','-i',str(args.identity),args.host]
        self.remote_root=args.project+'/data'
        self.local=ROOT/'runs/remote_migration'
        self.local.mkdir(parents=True,exist_ok=True)
        self.started=time.time()

    def status(self,phase,**values):
        record=dict(phase=phase,updated_at=datetime.now(timezone.utc).isoformat(),
                    elapsed_s=time.time()-self.started,remote_project=self.args.project,
                    remote_data_root=self.remote_root,local_training=False,music_only=True,
                    run_root=self.remote_root+'/runs/music',
                    stage_directories={'4':self.remote_root+'/runs/remote-stage4/stage4'},
                    log=self.remote_root+'/runs/music-preparation/'+phase+'.log',**values)
        write_json(self.local/'music_status.json',record)
        self.remote("python3 - <<'PY'\nimport pathlib\np=pathlib.Path('data/runs/music/status.json')\n"
                    "p.parent.mkdir(parents=True,exist_ok=True)\nt=p.with_suffix('.json.tmp')\n"
                    't.write_text('+repr(json.dumps(record,indent=2)+'\n')+')\nt.replace(p)\nPY\n')
        print(json.dumps(record),flush=True)

    def remote(self,script,attempts=3):
        prefix='set -e\ncd '+shlex.quote(self.args.project)+'\nexport SAMSONE_CLM_DATA='+shlex.quote(self.remote_root)+'\n'
        for attempt in range(attempts):
            result=subprocess.run(self.ssh+['bash','-s'],input=prefix+script,text=True,capture_output=True)
            if result.returncode==0:
                return result.stdout
            if result.returncode != 255 or attempt==attempts-1:
                raise RuntimeError(result.stderr[-4000:]+'\n'+result.stdout[-4000:])
            time.sleep(10)

    def get_json(self,path):
        output=self.remote('python3 - <<\'PY\'\nimport json,pathlib\np=pathlib.Path('+repr(path)+')\nprint(p.read_text() if p.exists() else "null")\nPY\n')
        return json.loads(output)

    def job(self,name,command):
        base='data/runs/music-preparation/'+name
        exit_path=base+'.exit.json'
        previous=self.get_json(exit_path)
        if previous and previous['exit_code']==0:
            self.status(name,status='already_completed')
            return
        state=self.remote('python3 - <<\'PY\'\nimport pathlib,os,json\np=pathlib.Path('+repr(base+'.pid')+')\n'
                         'pid=int(p.read_text()) if p.exists() else -1\n'
                         'proc=pathlib.Path("/proc",str(pid))\n'
                         'try: alive=pid>0 and proc.exists() and proc.joinpath("stat").read_text().split(")",1)[1].split()[0]!="Z" and str(proc.joinpath("cwd").resolve())==os.getcwd() and os.getpgid(pid)==pid\n'
                         'except (OSError,IndexError): alive=False\n'
                         'print(json.dumps(dict(alive=alive,pid=pid)))\nPY\n')
        alive=json.loads(state)['alive']
        if not alive:
            worker='set +e\n'+shlex.join(command)+'\ncode=$?\n'+\
                   'printf \'{"exit_code":%s}\\n\' "$code" > '+shlex.quote(exit_path+'.tmp')+'\n'+\
                   'mv '+shlex.quote(exit_path+'.tmp')+' '+shlex.quote(exit_path)+'\nexit "$code"\n'
            self.remote('mkdir -p data/runs/music-preparation\nrm -f '+shlex.quote(exit_path)+'\n'+
                        'nohup setsid bash -c '+shlex.quote(worker)+' >'+shlex.quote(base+'.log')+' 2>&1 </dev/null &\n'+
                        'printf \'%s\\n\' "$!" >'+shlex.quote(base+'.pid')+'\n')
        while True:
            result=self.get_json(exit_path)
            if result:
                self.status(name,status='completed' if result['exit_code']==0 else 'failed',exit_code=result['exit_code'])
                if result['exit_code']:
                    tail=self.remote('tail -n 30 '+shlex.quote(base+'.log')+'\n')
                    raise RuntimeError(name+' failed:\n'+tail[-6000:])
                return
            info=self.remote('python3 - <<\'PY\'\nimport pathlib,json,os\n'
                     'p=pathlib.Path('+repr(base+'.pid')+');pid=int(p.read_text())\n'
                     'r=pathlib.Path("data/runs/music-preparation/resources.json")\n'
                     'proc=pathlib.Path("/proc",str(pid))\n'
                     'try: alive=proc.exists() and proc.joinpath("stat").read_text().split(")",1)[1].split()[0]!="Z" and str(proc.joinpath("cwd").resolve())==os.getcwd() and os.getpgid(pid)==pid\n'
                     'except (OSError,IndexError): alive=False\n'
                     'print(json.dumps(dict(alive=alive,resources=json.loads(r.read_text()) if r.exists() else {})))\nPY\n')
            info=json.loads(info)
            if not info['alive']:
                # The wrapper may finish between the two SSH probes. Its exit
                # record is authoritative; a dead PID alone is not a failure.
                result=self.get_json(exit_path)
                if result:
                    if result['exit_code']:
                        raise RuntimeError(name+' failed with exit code '+str(result['exit_code']))
                    self.status(name,status='completed',exit_code=0)
                    return
                raise RuntimeError(name+' exited without publishing a status; inspect worker log')
            self.status(name,status='running',**info)
            time.sleep(30)

    def release_archive_cache(self):
        self.remote(".venv/bin/python - <<'PY'\nimport os,pathlib\nfrom samsone_clm.resources import require_resources\n"
                    "for p in pathlib.Path('data/raw/music').glob('*/*.tar.gz'):\n"
                    " with p.open('rb') as f:os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)\n"
                    "print(require_resources())\nPY\n")

    def precompute(self,dataset,batch):
        report=self.get_json('data/processed/music/'+dataset+'.report.json')
        if not report or report['split_leaks']:
            raise RuntimeError(dataset+' preparation is incomplete or has split leaks')
        self.release_archive_cache()
        self.job('precompute-'+dataset,['.venv/bin/python','-m','samsone_clm.precompute',
                 '--dataset','data/processed/music/'+dataset+'.jsonl','--batch-size',str(batch),
                 '--shard-size','512','--ablation-splits','validation','test',
                 '--ablation-limit-per-split','2048','--pointer','data/features/'+dataset+'_manifest.json'])
        return self.get_json('data/features/'+dataset+'_manifest.json')['path']

    def compose(self,stage,new):
        command=['.venv/bin/python','scripts/compose_music_features.py','--stage',str(stage),
                 '--new-manifests',*new,'--common-manifest','data/features/spike-a86e057eae695e43/manifest.json']
        if stage==6:
            command+=['--stage5-manifest',self.get_json('data/features/music_stage5_manifest.json')['path']]
        self.job('compose-stage'+str(stage),command)

    def train_stage(self,stage):
        self.release_archive_cache()
        self.job('overfit-stage'+str(stage),['.venv/bin/python','scripts/train_music.py','--stage',str(stage),'--diagnostic','--resume'])
        gate=self.get_json('data/runs/music/overfit_stage'+str(stage)+'_gate.json')
        if not gate or not gate['passed']:
            raise RuntimeError('Overfit gate failed for Stage '+str(stage))
        self.job('train-stage'+str(stage),['.venv/bin/python','scripts/train_music.py','--stage',str(stage),
                 '--batch-size','256','--epochs','32','--resume'])
        manifest=self.get_json('data/features/music_stage'+str(stage)+'_manifest.json')['path']
        run='data/runs/music/stage'+str(stage)
        self.job('finalize-stage'+str(stage),['.venv/bin/python','scripts/finalize_music_stage.py',
                 '--stage',str(stage),'--feature-manifest',manifest])
        self.return_stage(stage)

    def rsync(self,relative,destination,filters=()):
        destination.mkdir(parents=True,exist_ok=True)
        transport=shlex.join(self.ssh[:-1])
        source=self.args.host+':'+self.remote_root+'/'+relative.rstrip('/')+'/'
        command=['rsync','-az','--compress-choice=zstd','--compress-level=3','--partial',
                 *filters,'-e',transport,source,str(destination)+'/']
        for attempt in range(3):
            result=subprocess.run(command)
            if result.returncode==0:return
            time.sleep(10)
        raise RuntimeError('Artifact transfer failed: '+relative)

    def return_stage(self,stage):
        self.status('return-stage'+str(stage),status='transferring')
        relative='runs/music/stage'+str(stage)
        destination=ROOT/relative
        self.rsync(relative,destination)
        checksums=self.verify_checksums(destination)
        pointer=self.get_json(f'data/features/music_stage{stage}_manifest.json')
        directory=Path(pointer['path']).parent.name
        self.rsync('features/'+directory,ROOT/'features'/directory)
        write_json(ROOT/f'features/music_stage{stage}_manifest.json',pointer)
        from samsone_clm.precompute import load_features
        import torch
        load_features(ROOT/'features'/directory/'manifest.json',dtype=torch.bfloat16)
        # Retain successful and failed mechanical diagnostics, not just the
        # selected large-run weights. All of these workers have finished.
        diagnostic_names=json.loads(self.remote("python3 - <<'PY'\nimport pathlib,json\n"
                'print(json.dumps([p.name for p in pathlib.Path("data/runs/music").glob("diagnostic-stage'+str(stage)+'-*") if p.is_dir()]))\nPY\n'))
        for name in diagnostic_names:
            self.remote(".venv/bin/python - <<'PY'\nfrom pathlib import Path\nfrom samsone_clm.common import sha256,write_json\n"
                        'p=Path('+repr('data/runs/music/'+name)+')\n'
                        "write_json(p/'checksums.json',{f.name:sha256(f) for f in p.iterdir() if f.is_file() and f.name!='checksums.json'})\nPY\n")
            self.rsync('runs/music/'+name,ROOT/'runs/music'/name)
            self.verify_checksums(ROOT/'runs/music'/name)
        for suffix in ('gate','failed_gate'):
            name=f'overfit_stage{stage}_{suffix}.json'
            report=self.get_json('data/runs/music/'+name)
            if report:write_json(ROOT/'runs/music'/name,report)
        marker=dict(stage=stage,verified_locally=True,verified_files=len(checksums),
                    best_sha256=checksums['best.pt'],feature_manifest_sha256=pointer['sha256'],
                    local_directory=str(destination),timestamp=time.time())
        write_json(ROOT/f'runs/music/returned_stage{stage}.json',marker)
        self.remote("python3 - <<'PY'\nimport json,pathlib\np=pathlib.Path("+repr(f'data/runs/music/returned_stage{stage}.json')+
                    ')\np.write_text('+repr(json.dumps(marker,indent=2)+'\n')+')\nPY\n')
        self.status('return-stage'+str(stage),status='verified',**marker)

    @staticmethod
    def verify_checksums(destination):
        checksums=json.loads((destination/'checksums.json').read_text())
        for name,expected in checksums.items():
            h=hashlib.sha256()
            with (destination/name).open('rb') as handle:
                for block in iter(lambda:handle.read(2**20),b''):h.update(block)
            if h.hexdigest()!=expected:
                raise RuntimeError('Returned artifact checksum mismatch: '+name)
        return checksums

    def verify_music_data(self):
        from concurrent.futures import ThreadPoolExecutor
        from samsone_clm.common import read_rows,resolve_data_path,sha256
        files={}
        reports={}
        for name in ('nsynth','jamendo','musicbench'):
            path=ROOT/f'processed/music/{name}.jsonl'
            report=json.loads(path.with_suffix('.report.json').read_text())
            if sha256(path)!=report['sha256']:raise RuntimeError('Returned canonical QA changed: '+name)
            rows=read_rows(path)
            if len(rows)!=report['samples'] or report['split_leaks']:raise RuntimeError('Returned dataset audit failed: '+name)
            for row in rows:
                audio=resolve_data_path(row['audio_path']);expected=row['audio_sha256']
                if audio in files and files[audio]!=expected:raise RuntimeError('Conflicting returned audio hashes')
                files[audio]=expected
            reports[name]=dict(qa=len(rows),canonical_sha256=report['sha256'],split_leaks=0)
        def check(item):
            path,expected=item
            if sha256(path)!=expected:raise RuntimeError('Returned audio checksum mismatch: '+str(path))
        with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(check,files.items()))
        write_json(ROOT/'runs/music/returned_music_data.json',dict(verified=True,audio_files=len(files),datasets=reports))

    def run(self):
        self.status('starting',status='running')
        # Preparation was started separately; wait for its committed reports.
        deadline=time.monotonic()+1800
        while any(self.get_json('data/processed/music/'+name+'.report.json') is None for name in ('jamendo','nsynth','musicbench')):
            if time.monotonic()>deadline:raise RuntimeError('Music preparation did not finish within30minutes')
            self.status('preparation',status='waiting');time.sleep(30)
        benchmarks=self.get_json('data/runs/music-preparation/encoder_30s_benchmark.json')
        safe=[r for r in (benchmarks or {}).get('results',[]) if r.get('finite') and r['peak_allocated_gib']<18]
        batch=max(safe,key=lambda r:r['states_per_second'])['batch'] if safe else 16
        first=[self.precompute(name,batch) for name in ('nsynth','jamendo')]
        self.compose(5,first)
        self.train_stage(5)
        structural=self.precompute('musicbench',batch)
        self.compose(6,[structural])
        self.train_stage(6)
        self.status('return-model',status='transferring')
        self.rsync('models/samsone_clm_music',ROOT/'models/samsone_clm_music')
        self.verify_checksums(ROOT/'models/samsone_clm_music')
        self.status('return-music-data',status='transferring')
        self.rsync('processed/music',ROOT/'processed/music')
        self.status('verify-music-data',status='verifying')
        self.verify_music_data()
        # All prepared music audio and canonical QA are included above. Return
        # frozen feature stores too, preserving logical paths and raw hashes.
        for stage in (5,6):
            pointer=self.get_json(f'data/features/music_stage{stage}_manifest.json')
            directory=Path(pointer['path']).parent.name
            self.rsync('features/'+directory,ROOT/'features'/directory)
            write_json(ROOT/f'features/music_stage{stage}_manifest.json',pointer)
            # Loading performs the manifest's SHA checks without running Samsone.
            from samsone_clm.precompute import load_features
            load_features(ROOT/'features'/directory/'manifest.json')
        self.rsync('raw/music',ROOT/'raw/music',filters=['--prune-empty-dirs','--include=*/','--include=*.json',
                   '--include=*.jsonl','--include=*.tsv','--exclude=*'])
        self.rsync('runs/music-preparation',ROOT/'runs/music-preparation')
        self.status('completed',status='completed',stages=[5,6],model=str(ROOT/'models/samsone_clm_music'))
        self.remote("python3 - <<'PY'\nimport pathlib,json\np=pathlib.Path('data/runs/music/completed.json')\np.write_text(json.dumps({'completed':True,'stages':[5,6]}))\nPY\n")


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',required=True)
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--identity',type=Path,required=True)
    parser.add_argument('--project',default='/home/ubuntu/dev_clm')
    args=parser.parse_args()
    pipeline=Pipeline(args)
    try:
        pipeline.run()
    except BaseException as error:
        pipeline.status('failed',status='failed',error=str(error))
        raise
