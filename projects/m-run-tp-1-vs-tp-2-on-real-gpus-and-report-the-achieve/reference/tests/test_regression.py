import sys
sys.path.insert(0, ".")
from tpscaling.benchmark import run_benchmark
from tpscaling.analysis import compute_scaling_efficiency

def test_benchmark_outputs():
    cfg = {"requests": 50, "tokens_per_req": 64, "base_latency": 5.0}
    res1 = run_benchmark(1, cfg)
    res2 = run_benchmark(2, cfg)
    assert res1["tp_degree"] == 1
    assert res2["tp_degree"] == 2
    assert res1["throughput"] > 0
    assert res2["throughput"] > 0

def test_scaling_efficiency_bounds():
    res1 = {"throughput": 1000.0}
    res2 = {"throughput": 1800.0}
    metrics = compute_scaling_efficiency(res1, res2)
    assert metrics["throughput_ratio"] == 1.8
    assert metrics["scaling_efficiency"] == 0.9
    assert 0.0 <= metrics["scaling_efficiency"] <= 1.5
