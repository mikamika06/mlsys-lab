import sys
sys.path.insert(0, ".")
from evict.parser import reconstruct_eviction_order
from evict.unload import force_unload_and_verify
from evict.mmap_load import compare_mmap_loads

def test_eviction_order_monotonic():
    snaps = [["a", "b", "c"], ["a", "c"], ["a"]]
    order = reconstruct_eviction_order(snaps)
    assert len(order) == 2
    assert order == ["b", "c"]

def test_unload_verification():
    before = [{"id": "a", "status": "active"}]
    after = []
    assert force_unload_and_verify("a", before, after) is True

def test_mmap_efficiency():
    res = compare_mmap_loads()
    assert res.get("mmap_cheaper") is True
