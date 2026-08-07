import sys
sys.path.insert(0, ".")
import numpy as np
from quantcorr.budget import allocate_eora_ranks


def test_allocate_eora_ranks_within_budget():
    w1 = np.random.randn(64, 64)
    w2 = np.random.randn(64, 64)
    weights = [w1, w2]
    budget = 500
    ranks = allocate_eora_ranks(weights, budget, 4)
    total_params = sum(r * (w.shape[0] + w.shape[1]) for r, w in zip(ranks, weights))
    assert total_params <= budget, f"Allocated params {total_params} exceed budget {budget}"


def test_allocate_eora_ranks_non_negative():
    w1 = np.random.randn(32, 32)
    ranks = allocate_eora_ranks([w1], 100, 4)
    assert all(r >= 0 for r in ranks), "Ranks must be non-negative"
