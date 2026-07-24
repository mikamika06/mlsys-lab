import numpy as np
from mlsys.scorers import max_abs_err

def grade(sol, fx) -> dict:
    # Generate several random test cases
    rng = np.random.default_rng(seed=42)
    err_max = 0.0
    for _ in range(3):
        n_kv = rng.integers(1, 6)
        g = rng.integers(2, 5)
        d = rng.integers(4, 9)
        n_q = n_kv * g
        Q = rng.standard_normal((n_q, d)).astype(np.float64)
        K = rng.standard_normal((n_kv, d)).astype(np.float64)
        V = rng.standard_normal((n_kv, d)).astype(np.float64)

        # Reference implementation
        j_indices = np.arange(n_q) // g
        K_sel = K[j_indices]
        V_sel = V[j_indices]
        scores = np.sum(Q * K_sel, axis=1)
        O_ref = scores[:, None] * V_sel

        try:
            O_sol = sol.gqa_attention(Q, K, V, g)
        except Exception:
            return {"max_abs_err": float("inf")}

        err = max_abs_err(O_ref, O_sol)
        if err > err_max:
            err_max = err
    return {"max_abs_err": err_max}
