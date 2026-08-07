import sys
sys.path.insert(0, ".")

from engine_diag.cudagraph import compute_padded_tokens, optimize_buckets


def test_optimize_buckets_minimizes_waste():
    histogram = {1: 100, 4: 50, 8: 20, 15: 10}
    max_batch = 16
    k = 3

    best_buckets = optimize_buckets(k, max_batch, histogram)
    best_waste = compute_padded_tokens(best_buckets, histogram)

    suboptimal_buckets = [2, 8, 16]
    suboptimal_waste = compute_padded_tokens(suboptimal_buckets, histogram)

    assert best_waste <= suboptimal_waste
    assert max_batch in best_buckets
    assert len(best_buckets) == k
