import numpy as np
from mlsys.scorers import rel_err

def _reference(batch_size, seq_len, hidden_dim, ff_hidden_mult):
    b, s, h, f = batch_size, seq_len, hidden_dim, ff_hidden_mult
    flops = 6 * b * s * h ** 2 + 2 * b * s * f * h ** 2 + 2 * b * s ** 2 * h
    bytes_ = 8 * b * s * h * (1 + 3 + 2 * f)
    return float(flops / bytes_)

def grade(sol, fx) -> dict:
    cases = [
        (1, 10, 768, 4),
        (2, 20, 1024, 4),
        (4, 30, 2048, 4),
    ]
    for b, s, h, f in cases:
        try:
            got = sol.compute_hbm_utilization(b, s, h, f)
        except Exception:
            return {"rel_err": float("inf")}
        ref = _reference(b, s, h, f)
        err = rel_err(np.array([got]), np.array([ref]))
        if err > 1e-9:
            return {"rel_err": err}
    return {"rel_err": 0.0}
