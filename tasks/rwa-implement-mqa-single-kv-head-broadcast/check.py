import numpy as np
from mlsys.scorers import max_abs_err

def _oracle(Q, K, V):
    d_k = Q.shape[-1]
    scores = np.einsum('qhd,hd->qh', Q, K) / np.sqrt(d_k)
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights /= np.sum(weights, axis=-1, keepdims=True)
    return weights[..., None] * V[0]

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    err_max = 0.0
    for n_q in (2, 5):
        for h in (1, 3):
            for d_k in (4, 8):
                for d_v in (3, 6):
                    Q = rng.standard_normal((n_q, h, d_k))
                    K = rng.standard_normal((1, d_k))
                    V = rng.standard_normal((1, d_v))
                    try:
                        out = sol.mqa_single_kv_broadcast(Q, K, V)
                    except Exception:
                        return {"max_abs_err": float("inf")}
                    ref = _oracle(Q, K, V)
                    err = max_abs_err(ref, out)
                    if err > err_max:
                        err_max = err
    return {"max_abs_err": err_max}
