import itertools
from typing import Dict, List


def compute_padded_tokens(buckets: List[int], batch_histogram: Dict[int, int]) -> int:
    """Compute total padded tokens wasted for a bucket setup and batch histogram."""
    sorted_buckets = sorted(buckets)
    total_waste = 0
    for batch_size, count in batch_histogram.items():
        if count <= 0:
            continue
        suitable = [b for b in sorted_buckets if b >= batch_size]
        if not suitable:
            continue
        chosen_bucket = suitable[0]
        waste_per_batch = chosen_bucket - batch_size
        total_waste += waste_per_batch * count
    return total_waste


def optimize_buckets(k: int, max_batch: int, batch_histogram: Dict[int, int]) -> List[int]:
    """Find bucket list of size k (including max_batch) minimizing padded tokens."""
    if k <= 0 or max_batch <= 0:
        return []
    if k == 1:
        return [max_batch]

    candidates = list(range(1, max_batch))
    best_buckets = None
    min_waste = float("inf")

    for combo in itertools.combinations(candidates, k - 1):
        buckets = sorted(list(combo) + [max_batch])
        waste = compute_padded_tokens(buckets, batch_histogram)
        if waste < min_waste:
            min_waste = waste
            best_buckets = buckets

    return best_buckets if best_buckets is not None else [max_batch]
