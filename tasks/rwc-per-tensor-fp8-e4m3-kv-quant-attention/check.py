import numpy as np
from mlsys.scorers import max_abs_err

def _quantize_dequant(x):
    amax = np.max(np.abs(x))
    scale = amax / 448.0 if amax != 0 else 1.0
    q = np.round(x / scale)
    # Clip to the signed 8‑bit range that e4m3 can represent after rounding.
    q = np.clip(q, -127, 127).astype(np.int8)
    return q.astype(np.float32) * scale

def _oracle(Q, K, V):
    K_dq = _quantize_dequant(K)
    V_dq = _quantize_dequant(V)
    d_k = Q.shape[-1]
    scores = Q @ K_dq.T / np.sqrt(d_k)
    e = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn = e / np.sum(e, axis=-1, keepdims=True)
    return attn @ V_dq

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(5):
        n = rng.integers(2, 6)
        d = rng.integers(4, 9)
        Q = rng.standard_normal((n, d)).astype(np.float32)
        K = rng.standard_normal((n, d)).astype(np.float32)
        V = rng.standard_normal((n, d)).astype(np.float32)
        ref = _oracle(Q, K, V)
        try:
            got = sol.quantized_attention(Q, K, V)
        except Exception:
            return {"max_abs_err": float("inf")}
        err = max_abs_err(ref, got)
        errs.append(err)
    return {"max_abs_err": max(errs)}
