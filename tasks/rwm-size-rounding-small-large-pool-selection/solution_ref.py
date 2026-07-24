import math

# Constants matching a production CUDA caching allocator's two-pool design.
MIN_BLOCK_SIZE = 512                 # every request rounds up to a multiple of this
SMALL_SIZE = 1024 * 1024             # 1 MiB: requests <= this go to the small pool
SMALL_BUFFER = 2 * 1024 * 1024       # 2 MiB: segment size used by the small pool
LARGE_BUFFER = 20 * 1024 * 1024      # 20 MiB: segment size for "mid-sized" large requests
MIN_LARGE_ALLOC = 10 * 1024 * 1024   # 10 MiB: requests >= this round their own segment
ROUND_LARGE = 2 * 1024 * 1024        # 2 MiB: segment rounding granularity above MIN_LARGE_ALLOC


def _round_request(nbytes: int) -> int:
    """Round a raw byte request up to the nearest MIN_BLOCK_SIZE multiple."""
    n = max(int(nbytes), 1)
    return MIN_BLOCK_SIZE * math.ceil(n / MIN_BLOCK_SIZE)


def _segment_size(rounded: int) -> int:
    """Size of the underlying memory segment used to service a (rounded)
    request of this size."""
    if rounded <= SMALL_SIZE:
        return SMALL_BUFFER
    if rounded < MIN_LARGE_ALLOC:
        return LARGE_BUFFER
    return ROUND_LARGE * math.ceil(rounded / ROUND_LARGE)


def route_allocation(nbytes: int):
    """
    Route a raw allocation request through the size-rounding + pool-selection
    logic of a two-pool caching allocator.

    Returns (pool, segment_size):
      pool         : "small" if the ROUNDED request size is <= 1 MiB,
                     otherwise "large".
      segment_size : the size (in bytes) of the memory segment that would be
                      carved out / reserved to service this request.
    """
    rounded = _round_request(nbytes)
    pool = "small" if rounded <= SMALL_SIZE else "large"
    segment = _segment_size(rounded)
    return pool, segment
