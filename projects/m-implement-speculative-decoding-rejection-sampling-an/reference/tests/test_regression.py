import sys

sys.path.insert(0, ".")
from specdec.metrics import compute_speedup
from specdec.optimize import find_optimal_k
from specdec.sampling import rejection_sample


def test_find_optimal_k_varies_with_cost_ratio():
    best1, _ = find_optimal_k(alpha=0.9, c=0.01, max_k=10)
    best2, _ = find_optimal_k(alpha=0.9, c=0.5, max_k=10)
    assert best1 > best2, f"Higher cost ratio should lower optimal k, got {best1} vs {best2}"


def test_find_optimal_k_monotonic_with_acceptance():
    best_low, _ = find_optimal_k(alpha=0.2, c=0.1, max_k=10)
    best_high, _ = find_optimal_k(alpha=0.95, c=0.1, max_k=10)
    assert best_high >= best_low, f"Higher alpha should not decrease optimal k"


def test_speedup_bounds():
    res = compute_speedup([2, 2, 2], k=2, draft_cost=0.1, target_cost=1.0)
    assert res["realized_speedup"] > 1.0
    assert res["mean_acceptance_rate"] == 1.0
