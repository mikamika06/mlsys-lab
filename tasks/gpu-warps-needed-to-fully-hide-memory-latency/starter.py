import math

def min_warps_to_hide_latency(L: int, I: int) -> int:
    """Return the minimum number of resident warps to fully hide memory latency.

    Uses Little's Law: W_min = ceil(L / I)
    L: round-trip latency in cycles
    I: independent instructions per warp before stall
    """
    raise NotImplementedError("your code here")
