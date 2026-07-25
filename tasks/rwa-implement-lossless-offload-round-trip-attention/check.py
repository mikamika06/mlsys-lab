import numpy as np
from mlsys import scorers

def _reference_attention(q, k, v):
    d_k = q.shape[-1]
    scale = np.sqrt(d_k)
    logits = np.matmul(q, k.transpose(0, 2, 1)) / scale
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    return np.matmul(probs, v)

def grade(sol, fx) -> dict:
    # seeded, or the same submission scores differently on every run
    rng = np.random.default_rng(0)
    q = rng.standard_normal((2, 3, 4))
    k = rng.standard_normal((2, 5, 4))
    v = rng.standard_normal((2, 5, 6))

    ref = _reference_attention(q, k, v)
    try:
        got = sol.offload_attention(q, k, v)
    except Exception:
        return {"max_abs_err": float("inf")}

    got = np.asarray(got)
    if got.shape != ref.shape:
        return {"max_abs_err": float("inf")}
    return {"max_abs_err": float(scorers.max_abs_err(ref, got))}
