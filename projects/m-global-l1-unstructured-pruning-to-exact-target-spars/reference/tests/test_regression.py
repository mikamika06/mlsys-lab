import sys
sys.path.insert(0, ".")
from prune.alloc import allocate_sparsity

def test_allocation_reaches_target():
    layers = [{"name": "l1", "shape": (32, 32)}, {"name": "l2", "shape": (64, 32)}]
    res = allocate_sparsity(layers, 0.5, "uniform")
    assert len(res) == 2
    assert res["l1"] == 0.5
    assert res["l2"] == 0.5
