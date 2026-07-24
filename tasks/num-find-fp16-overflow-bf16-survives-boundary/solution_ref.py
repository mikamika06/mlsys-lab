import numpy as np

def find_fp16_overflow_boundary() -> float:
    """
    Return the smallest positive real number x such that
    np.float16(x) == inf but np.bfloat16(x) != inf.
    Uses a binary search between 65504 and an upper bound.
    If np.bfloat16 is unavailable, falls back to np.float32.
    """
    low = 65504.0
    high = 70000.0

    try:
        cast_bf = np.bfloat16
    except AttributeError:
        cast_bf = np.float32

    while high - low > 1e-6:
        mid = (low + high) / 2.0
        if np.float16(mid) == np.inf and cast_bf(mid) != np.inf:
            high = mid
        else:
            low = mid

    return float(high)
