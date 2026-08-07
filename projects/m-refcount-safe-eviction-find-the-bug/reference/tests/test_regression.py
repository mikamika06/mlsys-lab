import sys

sys.path.insert(0, ".")
from radixkv.eviction import simulate_eviction
from radixkv.overhead import tree_memory_overhead
from radixkv.fork import fork_tokens_saved

def test_eviction_zero_refcount():
    tree = {
        "nodes": [
            {"id": 1, "parent": None, "refcount": 1, "children": [2]},
            {"id": 2, "parent": 1, "refcount": 1, "children": []}
        ]
    }
    evicted = simulate_eviction(tree, 2)
    assert 2 in evicted

def test_memory_overhead_positive():
    val = tree_memory_overhead(100, 4, 64)
    assert val > 0

def test_fork_tokens_saved():
    saved = fork_tokens_saved(100, 4, 32)
    assert saved == 96
