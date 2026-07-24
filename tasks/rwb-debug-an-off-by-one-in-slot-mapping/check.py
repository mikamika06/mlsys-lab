import numpy as np


def _oracle_attention(K, V, Q):
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    T, H, D = Q.shape
    out = np.empty((T, H, D), dtype=np.float64)
    scale = 1.0 / np.sqrt(D)
    for i in range(T):
        scores = np.einsum("hd,lhd->hl", Q[i], K) * scale
        scores -= np.max(scores, axis=1, keepdims=True)
        probs = np.exp(scores)
        probs /= np.sum(probs, axis=1, keepdims=True)
        out[i] = np.einsum("hl,lhd->hd", probs, V)
    return out


def _make_case(seed, T, B, H, D):
    rng = np.random.default_rng(seed)
    K = rng.normal(size=(T, H, D))
    V = rng.normal(size=(T, H, D))
    Q = rng.normal(size=(T, H, D))
    nb = (T + B - 1) // B
    Kc = np.zeros((nb, B, H, D), dtype=np.float64)
    Vc = np.zeros((nb, B, H, D), dtype=np.float64)
    for p in range(T):
        Kc[p // B, p % B] = K[p]
        Vc[p // B, p % B] = V[p]
    return Kc, Vc, Q, np.arange(T, dtype=np.int64), B, K, V


def grade(sol, fx) -> dict:
    cases = [
        _make_case(7, 12, 4, 2, 5),
        _make_case(11, 15, 3, 3, 4),
        _make_case(19, 9, 5, 1, 6),
    ]
    best = 0.0
    for Kc, Vc, Q, positions, B, K, V in cases:
        try:
            got = sol.slot_attention(Kc, Vc, Q, positions, B)
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = _oracle_attention(K, V, Q)
        err = float(np.max(np.abs(np.asarray(got, dtype=np.float64) - ref)))
        best = max(best, err)
    return {"max_abs_err": best}
