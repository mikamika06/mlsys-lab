import itertools
import re
from typing import Dict, List, Tuple


LOG_SAMPLES = [
    """
    [2026-08-07 10:00:00] INFO [PHASE:init_weights] START=0.00 END=2.35
    [2026-08-07 10:00:02] INFO [PHASE:allocate_kv_cache] START=2.35 END=5.10
    [2026-08-07 10:00:05] INFO [PHASE:compile_cudagraphs] START=5.10 END=18.40
    [2026-08-07 10:00:18] INFO [PHASE:warmup_engine] START=18.40 END=20.00
    """,
    """
    [PHASE:model_loading] START=100.0 END=104.5
    [PHASE:cudagraph_capture] START=104.5 END=112.2
    """,
    """
    DEBUG: starting
    [PHASE:p0] START=10.0 END=10.1
    [PHASE:p1] START=10.1 END=12.5
    [PHASE:p2] START=12.5 END=13.0
    DEBUG: finished
    """
]


BUCKET_TEST_CASES = [
    {
        "k": 3,
        "max_batch": 16,
        "histogram": {1: 80, 2: 50, 5: 30, 8: 20, 14: 10}
    },
    {
        "k": 2,
        "max_batch": 8,
        "histogram": {1: 10, 3: 20, 7: 5}
    },
    {
        "k": 4,
        "max_batch": 12,
        "histogram": {2: 15, 4: 25, 6: 10, 11: 5}
    }
]


def parse_startup_log(log_text: str) -> Dict[str, float]:
    phase_times = {}
    pattern = re.compile(r"\[PHASE:([a-zA-Z0-9_]+)\]\s+START=(\d+\.?\d*)\s+END=(\d+\.?\d*)")
    for line in log_text.strip().splitlines():
        match = pattern.search(line)
        if match:
            phase_name, start_s, end_s = match.groups()
            duration = float(end_s) - float(start_s)
            phase_times[phase_name] = round(duration, 4)
    return phase_times


def total_startup_time(phase_breakdown: Dict[str, float]) -> float:
    return round(sum(phase_breakdown.values()), 4)


def compute_padded_tokens(buckets: List[int], batch_histogram: Dict[int, int]) -> int:
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
