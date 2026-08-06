import sys

sys.path.insert(0, ".")
from enginebench.metrics import compute_throughput, compute_throughput_ratio
from enginebench.runner import parse_config


def test_parse_config_valid():
    cfg = {"model": "test-model", "prompts": ["hello"], "concurrency": 8, "engine": "engine_a"}
    parsed = parse_config(cfg)
    assert parsed["concurrency"] == 8
    assert parsed["engine"] == "engine_a"


def test_throughput_calculation_positive():
    latencies = [10.0, 12.0, 15.0]
    tp = compute_throughput(latencies, 300)
    assert tp > 0.0


def test_throughput_ratio_bounded():
    base = [10.0, 11.0]
    cand = [8.0, 9.0]
    ratio = compute_throughput_ratio(base, cand, 100, 1)
    assert ratio > 1.0
