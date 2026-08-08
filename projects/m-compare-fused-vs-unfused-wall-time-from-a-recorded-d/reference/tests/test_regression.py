import sys
sys.path.insert(0, ".")
from triton_bench.analysis import parse_benchmark, compute_ratios
from triton_bench.metrics import evaluate_throughput


def test_parse_valid_data():
    raw = [{"name": "fused", "size": 1024, "time_ms": 1.5}]
    parsed = parse_benchmark(raw)
    assert len(parsed) == 1
    assert parsed[0]["size"] == 1024


def test_compute_ratios_positive():
    r = compute_ratios([1.0, 2.0], [2.0, 4.0])
    assert all(x == 2.0 for x in r)


def test_evaluate_throughput_threshold():
    val = evaluate_throughput([1.0, 1.0], [2.0, 2.0])
    assert val >= 1.2
