import sys

sys.path.insert(0, ".")
from specpair.metrics import compute_acceptance_rate
from specpair.selector import compute_expected_throughput, select_optimal_draft
from specpair.tokenizer import classify_tokenizer_compatibility


def test_acceptance_rate_bounds():
    assert compute_acceptance_rate([1, 2, 3], [1, 2, 3, 4]) == 1.0
    assert compute_acceptance_rate([1, 2, 3], [1, 4, 3]) == 1 / 3
    assert compute_acceptance_rate([1, 2, 3], [5, 6, 7]) == 0.0


def test_tokenizer_compatibility_classification():
    v1 = ["a", "b", "c"]
    v2 = ["a", "b", "c"]
    v3 = ["a", "b", "c", "d"]
    v4 = ["x", "y"]
    assert classify_tokenizer_compatibility(v1, v2) == "identical"
    assert classify_tokenizer_compatibility(v1, v3) == "compatible_subset"
    assert classify_tokenizer_compatibility(v1, v4) == "cross_tokenizer"


def test_throughput_selection():
    candidates = [
        {"name": "d1", "draft_latency": 2.0, "acceptance_rate": 0.8},
        {"name": "d2", "draft_latency": 1.0, "acceptance_rate": 0.5},
    ]
    res = select_optimal_draft(candidates, target_latency=20.0, gamma=5)
    assert res["best_draft"] in ("d1", "d2")
    assert res["throughput"] > 0.0


def test_zero_acceptance_behavior():
    tp = compute_expected_throughput(
        gamma=5, acceptance_rate=0.0, draft_latency=1.0, target_latency=10.0
    )
    assert abs(tp - 1.0 / 15.0) < 1e-6
