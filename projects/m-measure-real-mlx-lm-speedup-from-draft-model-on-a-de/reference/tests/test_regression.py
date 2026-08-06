import sys
sys.path.insert(0, ".")
from spec.moe_regression import evaluate_moe_speculative


def test_moe_speculative_regression_is_caught():
    res = evaluate_moe_speculative({"target": "moe-8x7b", "draft": "dense-1b"})
    assert res.get("moe_speedup", 1.0) < 1.0, f"Expected speedup regression on MoE target, got {res.get('moe_speedup')}"
    assert res.get("regression_detected") is True, "Expected regression flag to be True"
