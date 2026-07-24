import numpy as np
from mlsys.scorers import rel_err

def _ref(p, q, beta):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    beta = float(beta)
    m = beta * p + (1 - beta) * q
    eps = 1e-12
    kl_p_m = np.sum(np.where(p > 0, p * np.log(p / (m + eps)), 0.0))
    kl_q_m = np.sum(np.where(q > 0, q * np.log(q / (m + eps)), 0.0))
    return beta * kl_p_m + (1 - beta) * kl_q_m

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    max_err = 0.0
    for _ in range(10):
        n = rng.integers(2, 20)
        p_raw = rng.random(n)
        q_raw = rng.random(n)
        p = p_raw / p_raw.sum()
        q = q_raw / q_raw.sum()
        beta = rng.random()
        try:
            got = sol.generalized_jsd(p, q, beta)
        except Exception:
            return {"rel_err": 1.0}
        ref = _ref(p, q, beta)
        err = rel_err(np.array([got]), np.array([ref]))
        if err > max_err:
            max_err = float(err)
    return {"rel_err": max_err}
