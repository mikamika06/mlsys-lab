import sys

sys.path.insert(0, ".")
from slobudget.budget import compute_breakdown
from slobudget.optimize import find_optimal_batch


def test_breakdown_total_matches_sum():
    res = compute_breakdown(100.0, 10, 20.0, 2.0, 15)
    s = res["queue"] + res["compile"] + res["compute"]
    assert abs(res["total"] - s) < 1e-5


def test_optimal_batch_positive():
    b = find_optimal_batch(100.0, 10.0, 2.0, 32)
    assert b > 0


def test_breakdown_components_non_negative():
    res = compute_breakdown(100.0, 5, 10.0, 1.0, 10)
    assert res["queue"] >= 0
    assert res["compile"] >= 0
    assert res["compute"] >= 0
