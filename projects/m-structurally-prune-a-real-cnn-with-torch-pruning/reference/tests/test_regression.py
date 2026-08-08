import sys
sys.path.insert(0, ".")
from prune_net.toy_dep import propagate_toy_dependencies
from prune_net.cnn_pruner import prune_real_cnn
from prune_net.speedup import measure_speedup_gap

def test_toy_dep_bounds():
    res = propagate_toy_dependencies(10, [0, 1], 0.2)
    assert len(res) == 8

def test_cnn_pruner_non_empty():
    profile = {"layers": [{"name": "conv1", "out_channels": 32}]}
    res = prune_real_cnn(profile, 0.5)
    assert len(res["pruned_layers"]) == 1

def test_speedup_calculation():
    d = {"latency_ms": 100.0}
    p = {"latency_ms": 50.0, "sparsity_ratio": 0.5}
    res = measure_speedup_gap(d, p)
    assert res["real_speedup"] == 2.0
