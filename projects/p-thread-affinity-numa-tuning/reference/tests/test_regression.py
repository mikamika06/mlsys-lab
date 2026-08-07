import sys
from numa_tuning import affinity


def test_affinity_pinning():
    res = affinity.apply_pinning(0, 2)
    assert res.get("status") == "success", f"Pinning failed: {res}"


def test_numa_allocation_locality():
    mem = affinity.allocate_numa_memory(512, 1)
    assert mem.get("allocated") is True, f"NUMA allocation failed: {mem}"
