import numpy as np
from mlsys.scorers import rel_err

def _ref_slopes(n):
    """Reference implementation using the closed‑form algorithm."""
    def get_power_of_two_slopes(k):
        start = 2 ** (-(k + 3))
        ratio = 2.0
        return np.array([start * (ratio ** i) for i in range(k)], dtype=np.float32)

    if n & (n - 1) == 0:          # power of two
        return get_power_of_two_slopes(n)
    else:
        k = 2 ** int(np.floor(np.log2(n)))
        slopes = get_power_of_two_slopes(k).tolist()
        extra = _ref_slopes(n - k).tolist()
        slopes.extend(extra)
        return np.array(slopes, dtype=np.float32)

def grade(sol, fx) -> dict:
    try:
        got = sol.alibi_slopes(13)
        ref = _ref_slopes(13)
    except Exception:
        return {"rel_err": 1.0}
    err = rel_err(ref, got)
    return {"rel_err": err}
