from samsone_clm.resources import cgroup_memory_available


def test_container_memory_limit_includes_reclaimable_file_cache(tmp_path):
    (tmp_path/'memory.max').write_text('1000\n')
    (tmp_path/'memory.current').write_text('800\n')
    (tmp_path/'memory.stat').write_text('inactive_file 300\nactive_file 100\n')
    assert cgroup_memory_available(tmp_path) == (500, 1000)
    (tmp_path/'memory.max').write_text('max\n')
    assert cgroup_memory_available(tmp_path) == (None, None)


def test_cgroup_v1_memory_headroom_never_uses_host_total(tmp_path):
    memory = tmp_path/'memory'
    memory.mkdir()
    (memory/'memory.limit_in_bytes').write_text('1000\n')
    (memory/'memory.usage_in_bytes').write_text('1200\n')
    (memory/'memory.stat').write_text('total_inactive_file 100\n')
    assert cgroup_memory_available(tmp_path) == (0, 1000)
