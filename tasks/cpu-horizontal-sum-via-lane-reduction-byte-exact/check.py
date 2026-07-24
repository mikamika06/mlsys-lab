import numpy as np
from mlsys import scorers

def grade(sol, fx) -> dict:
    tests = [
        np.array([1, 2, 3, 4], dtype=np.int32),
        np.zeros(0, dtype=np.int32),
        np.arange(-10, 10, dtype=np.int32),
        np.random.randint(-1000, 1000, size=1024, dtype=np.int32)
    ]
    for arr in tests:
        try:
            got = sol.lane_reduce_sum(arr)
        except Exception:
            return {"byte_exact_fraction": 0.0}
        expected = np.array(np.sum(arr), dtype=arr.dtype)
        frac = scorers.byte_exact_fraction(expected, got)
        if frac < 1.0:
            return {"byte_exact_fraction": 0.0}
    return {"byte_exact_fraction": 1.0}
