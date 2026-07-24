"""Oracle: standard causal multi-step, multi-head, multi-layer attention
computed directly from the full ("everything stays on device") K/V history,
vectorised with NumPy. This is the ground truth that a CPU-offloaded KV
cache -- which only relocates storage, never changes values -- must
reproduce exactly (up to float64 rounding) at every decode step and layer.
"""
import numpy as np


def _full_cache_attention(Q, K, V):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    L, T, H, d = Q.shape
    scale = 1.0 / np.sqrt(d)
    out = np.zeros_like(Q)
    causal = np.triu(np.ones((T, T), dtype=bool), k=1)  # True = future, masked out
    for l in range(L):
        for h in range(H):
            q = Q[l, :, h, :]
            k = K[l, :, h, :]
            v = V[l, :, h, :]
            scores = (q @ k.T) * scale
            scores = np.where(causal, -np.inf, scores)
            scores = scores - np.max(scores, axis=-1, keepdims=True)
            w = np.exp(scores)
            w = w / np.sum(w, axis=-1, keepdims=True)
            out[l, :, h, :] = w @ v
    return out


def _cases():
    return [
        dict(L=3, T=5, H=2, d=4, seed=0),
        dict(L=2, T=8, H=3, d=5, seed=1),
        dict(L=1, T=1, H=1, d=6, seed=2),   # single decode step, single layer/head
        dict(L=4, T=6, H=1, d=3, seed=3),
    ]


def grade(sol, fx) -> dict:
    worst = 0.0
    for c in _cases():
        rng = np.random.default_rng(c["seed"])
        L, T, H, d = c["L"], c["T"], c["H"], c["d"]
        Q = rng.standard_normal((L, T, H, d))
        K = rng.standard_normal((L, T, H, d))
        V = rng.standard_normal((L, T, H, d))

        ref = _full_cache_attention(Q, K, V)

        try:
            got = np.asarray(
                sol.offloaded_decode_attention(Q.copy(), K.copy(), V.copy()),
                dtype=np.float64,
            )
        except Exception:
            return {"max_abs_err": float("inf")}

        if got.shape != ref.shape or not np.all(np.isfinite(got)):
            return {"max_abs_err": float("inf")}

        worst = max(worst, float(np.max(np.abs(got - ref))))

    return {"max_abs_err": worst}
