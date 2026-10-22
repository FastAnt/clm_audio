"""Read-only telemetry for an additional Stage6 run while base artifacts return."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time

from samsone_clm.common import ROOT,write_json
from samsone_clm.monitor import read_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=ROOT/'runs/music/stage6-full32')
    parser.add_argument('--pipeline-status',type=Path)
    parser.add_argument('--completion-marker',type=Path,default=ROOT/'runs/music/full32_completed.json')
    args=parser.parse_args()
    while True:
        metadata=read_json(args.pipeline_status) if args.pipeline_status else {}
        report=read_json(args.run_dir/'metrics.json')
        job=metadata.get('job_name','train-stage6-full32')
        exit_status=read_json(ROOT/f'runs/music-preparation/{job}.exit.json')
        complete=args.completion_marker.exists()
        status='completed' if complete else 'failed' if exit_status.get('exit_code',0) else 'running'
        phase='completed' if complete else 'failed' if status=='failed' else metadata.get('phase') or ('finalize-stage6' if report.get('status')=='completed' else 'train-stage6')
        record=dict(phase=phase,status=status,updated_at=datetime.now(timezone.utc).isoformat(),
                    run_root=str(args.run_dir.parent),remote_data_root=str(ROOT),music_only=True,local_training=False,
                    stage_directories={'4':str(ROOT/'runs/remote-stage4/stage4'),'5':str(ROOT/'runs/music/stage5'),'6':str(args.run_dir)},
                    log=metadata.get('log',str(ROOT/f'runs/music-preparation/{job}.log')),resources=read_json(ROOT/'runs/music-preparation/resources.json'),
                    experiment=metadata.get('experiment','Stage6 full32; same features, parent, objective and LR; patience0'),
                    original_stage6_completed=True,artifact_return=read_json(ROOT/'runs/music/status.json').get('phase'))
        write_json(ROOT/'runs/music/refinement_status.json',record)
        if complete or status=='failed':return
        time.sleep(15)


if __name__=='__main__':main()
