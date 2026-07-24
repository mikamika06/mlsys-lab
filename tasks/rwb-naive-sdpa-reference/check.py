import numpy as np
from mlsys.scorers import max_abs_err

def _ref(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    d_k = Q.shape[-1]
    logits = (Q @ K.T) / np.sqrt(d_k)
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    softmax = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    out = softmax @ V
    return out.astype(Q.dtype)

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    max_err = 0.0
    for batch in [2, 5]:
        for d_k in [4, 8]:
            for d_v in [3, 6]:
                Q = rng.standard_normal((batch, d_k)).astype(np.float32)
                K = rng.standard_normal((d_k, d_k)).astype(np.float32)
                V = rng.standard_normal((d_k, d_v)).astype(np.float32)
                try:
                    got = sol.sdpa(Q, K, V)
                except Exception:
                    return {"max_abs_err": float("inf")}
                ref = _ref(Q, K, V)
                err = max_abs_err(ref, got)
                if err > max_err:
                    max_err = err
    return {"max_abs_err": max_err}
