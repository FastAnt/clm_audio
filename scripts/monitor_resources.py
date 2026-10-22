"""Monitor container RAM/disk/GPU and stop only verified project worker groups."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from samsone_clm.common import PROJECT, ROOT, write_json
from samsone_clm.resources import resource_status


def verified_workers():
    result = []
    for directory in (ROOT/'runs/music-preparation',ROOT/'runs/music',ROOT/'runs/music-genre'):
        for path in directory.glob('*.pid'):
            try:
                pid = int(path.read_text().strip())
                process = Path('/proc')/str(pid)
                cwd = Path(os.readlink(process/'cwd')).resolve()
                command = (process/'cmdline').read_bytes().decode(errors='replace')
                if cwd == PROJECT.resolve() and any(name in command for name in
                    ('download_music.py','prepare_music.py','train_music.py','samsone_clm.precompute','run_music_precompute.py','run_music_window_pipeline.py','download_genre_data.py','prepare_genre_data.py','run_genre_pipeline.py')):
                    if os.getpgid(pid) == pid:
                        result.append(pid)
            except (ValueError,FileNotFoundError,ProcessLookupError,PermissionError):
                continue
    return sorted(set(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interval',type=float,default=15)
    parser.add_argument('--min-ram-gib',type=float,default=15)
    parser.add_argument('--min-disk-gib',type=float,default=20)
    parser.add_argument('--completion-marker',type=Path,default=ROOT/'runs/music/completed.json')
    args = parser.parse_args()
    while True:
        status = resource_status()
        workers = verified_workers()
        status.update(timestamp=time.time(),verified_worker_pids=workers)
        gpu = subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu',
                              '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
        status['gpu_memory_used_total_mib_utilization_percent'] = gpu.stdout.strip()
        low = status['available_ram_gib'] < args.min_ram_gib or status['free_disk_gib'] < args.min_disk_gib
        status['status'] = 'resource_stop' if low else 'healthy'
        if low:
            for pid in workers:
                try:
                    os.killpg(pid,signal.SIGTERM)
                except ProcessLookupError:
                    pass
        write_json(ROOT/'runs/music-preparation/resources.json',status)
        print(json.dumps(status),flush=True)
        if low or args.completion_marker.exists():
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
