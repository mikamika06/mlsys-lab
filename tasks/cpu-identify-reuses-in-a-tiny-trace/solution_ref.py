import numpy as np
from mlsys.sim import cache as cachesim

def count_cache_hits(trace: np.ndarray) -> int:
    """
    Return the number of cache hits for a fixed LRU cache configuration.
    """
    result = cachesim.simulate(
        trace,
        line_bytes=8,
        sets=4,
        ways=2
    )
    return int(result['hits'])
