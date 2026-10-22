"""Resource checks for the unified-memory workstation; no disk-wide repair actions."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from .common import ROOT


def cgroup_memory_available(directory=Path('/sys/fs/cgroup')):
    """Container memory headroom; /proc/meminfo can report the whole host."""
    directory = Path(directory)
    for limit_path, usage_path, stat_path, inactive_key in (
        (directory/'memory.max', directory/'memory.current', directory/'memory.stat', 'inactive_file'),
        (directory/'memory/memory.limit_in_bytes', directory/'memory/memory.usage_in_bytes',
         directory/'memory/memory.stat', 'total_inactive_file'),
    ):
        if not limit_path.exists() or not usage_path.exists():
            continue
        value = limit_path.read_text().strip()
        if value == 'max':
            continue
        limit = int(value)
        if limit >= 2**60:
            continue
        usage = int(usage_path.read_text().strip())
        stats = dict(line.split() for line in stat_path.read_text().splitlines()) if stat_path.exists() else {}
        reclaimable = int(stats.get(inactive_key, 0))
        return max(0, min(limit, limit - usage + reclaimable)), limit
    return None, None


def resource_status():
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        memory[key] = int(value.strip().split()[0]) * 1024
    disk = shutil.disk_usage(ROOT)
    available, limit = cgroup_memory_available()
    headroom = min(memory['MemAvailable'], available) if available is not None else memory['MemAvailable']
    return dict(available_ram_gib=headroom/2**30,
                memory_limit_gib=limit/2**30 if limit is not None else None,
                free_disk_gib=disk.free/2**30,
                swap_used_gib=(memory['SwapTotal']-memory['SwapFree'])/2**30)


def require_resources(min_ram_gib=20., min_disk_gib=20.):
    status = resource_status()
    if status['available_ram_gib'] < min_ram_gib or status['free_disk_gib'] < min_disk_gib:
        raise RuntimeError(f'Insufficient resources: {status}')
    return status


def probe_storage(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'.storage-probe-{os.getpid()}'
    expected = b'samsone-clm-read-write-probe\n'
    with path.open('xb') as handle:
        handle.write(expected)
        handle.flush()
        os.fsync(handle.fileno())
    if path.read_bytes() != expected:
        raise RuntimeError('Storage probe read mismatch')
    path.unlink()
    return require_resources()
