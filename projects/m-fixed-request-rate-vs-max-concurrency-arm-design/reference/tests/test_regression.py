import sys
sys.path.insert(0, ".")
from bench.parser import parse_vllm_json
from bench.stats import compute_repeats_for_claim
from bench.compare import build_comparison_table

def test_parser_extracts_correct_fields():
    raw = '{"duration": 10.0, "completed": 100, "request_throughput": 10.0, "mean_latency_ms": 50.0, "p50_latency_ms": 45.0, "p99_latency_ms": 120.0}'
    res = parse_vllm_json(raw)
    assert res["completed"] == 100
    assert res["p99_latency_ms"] == 120.0

def test_stats_returns_positive_integer():
    lats = [50.0, 52.0, 48.0, 51.0, 49.0]
    n = compute_repeats_for_claim(lats, 0.05)
    assert isinstance(n, int)
    assert n >= 1

def test_comparison_table_ratio_positive():
    fr = [{"request_throughput": 10.0, "p99_latency_ms": 100.0}]
    mc = [{"request_throughput": 15.0, "p99_latency_ms": 150.0}]
    tbl = build_comparison_table(fr, mc)
    assert tbl["throughput_ratio"] > 1.0
    assert tbl["max_concurrency_mean_throughput"] == 15.0
