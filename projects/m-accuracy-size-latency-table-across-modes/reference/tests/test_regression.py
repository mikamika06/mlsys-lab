import sys
sys.path.insert(0, ".")
from modetbl.analyzer import build_profiles
from modetbl.metrics import compute_size_ratios, evaluate_tradeoffs

SAMPLE = [
    {"mode": "FP32", "size_bytes": 40000000, "latency_ms": 25.0, "accuracy": 0.785},
    {"mode": "INT8_FULL", "size_bytes": 10000000, "latency_ms": 11.0, "accuracy": 0.768},
]


def test_size_ratios_less_than_one():
    profiles = build_profiles(SAMPLE)
    ratios = compute_size_ratios(profiles)
    assert ratios["INT8_FULL"] < 1.0


def test_evaluate_tradeoffs_valid():
    profiles = build_profiles(SAMPLE)
    res = evaluate_tradeoffs(profiles)
    assert res["valid"] is True
