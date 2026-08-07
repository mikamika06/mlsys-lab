from typing import List, Tuple
import numpy as np
from lanewaste.metrics import calculate_wasted_lane_time

def select_best_block_size(n: int, candidates: List[int], launch_overhead: float) -> Tuple[int, float]:
    """Finds the candidate block size index minimizing total wasted lane time."""
    wastes = [calculate_wasted_lane_time(n, b, launch_overhead) for b in candidates]
    best_idx = int(np.argmin(wastes))
    return best_idx, wastes[best_idx]
