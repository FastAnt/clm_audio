"""Observe only the v3 worker; stop its own process group on resource exhaustion or a stall."""
import json
import os
import signal
import time
from pathlib import Path
from datetime import datetime,timezone

from samsone_clm.common import ROOT,write_json
from samsone_clm.resources import resource_status


def main():
    directory=ROOT/'runs/v3'
    previous=None
    last_change=time.monotonic()
    with (directory/'resource_history.jsonl').open('a') as log:
        while True:
            status=json.loads((directory/'status.json').read_text())
            resources=resource_status()
            record=dict(time=datetime.now(timezone.utc).isoformat(),phase=status.get('phase'),**resources)
            log.write(json.dumps(record)+'\n');log.flush()
            if status.get('status') in ('completed','failed'):
                return
            pid=status.get('active_pid')
            path=Path(status.get('log',str(directory/'supervisor.log')))
            try: stamp=(pid,str(path),path.stat().st_size)
            except OSError:stamp=(pid,str(path),None)
            if stamp!=previous:
                previous=stamp
                last_change=time.monotonic()
            reason=None
            if resources['available_ram_gib']<12 or resources['free_disk_gib']<12:
                reason='resource threshold breached'
            if pid and time.monotonic()-last_change>600:
                reason='worker produced no log progress for 600 seconds'
            if reason and pid:
                try:
                    command=Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\x00',' ')
                    if 'samsone-clm' not in command or os.getpgid(pid)!=pid:
                        raise RuntimeError('Refusing to signal an unverified worker process group')
                    write_json(directory/'watchdog_stop.json',dict(reason=reason,pid=pid,command=command,**record))
                    os.killpg(pid,signal.SIGTERM)
                    print(json.dumps(dict(stopped_pid=pid,reason=reason,resources=resources)),flush=True)
                    return
                except ProcessLookupError:
                    pass
            time.sleep(15)


if __name__=='__main__':main()
