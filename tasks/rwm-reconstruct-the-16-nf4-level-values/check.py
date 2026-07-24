import numpy as np
from mlsys.scorers import max_abs_err

def _oracle_nf4_levels():
    """Compute the 16 NF4 level values programmatically from normal quantiles."""
    from scipy.stats import norm
    # 16 quantiles of N(0,1)
    q = np.array([norm.ppf((i + 0.5) / 16) for i in range(16)])
    # Normalize to [-1, 1]
    q_norm = q / np.max(np.abs(q))
    # Force the entry closest to 0 to exactly 0.0
    k = np.argmin(np.abs(q_norm))
    q_norm[k] = 0.0
    return q_norm.astype(np.float64)

def grade(sol, fx) -> dict:
    try:
        result = sol.nf4_levels()
    except Exception:
        return {"max_abs_err": 1.0}  # fails the <= 1e-4 gate
    ref = _oracle_nf4_levels()
    err = max_abs_err(ref, result)
    return {"max_abs_err": err}
