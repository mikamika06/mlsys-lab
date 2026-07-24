import numpy as np


def _attention_loss(Q, K, V, B, dO):
    d = Q.shape[1]
    S = Q @ K.T / np.sqrt(float(d)) + B
    S = S - np.max(S, axis=1, keepdims=True)
    P = np.exp(S)
    P = P / np.sum(P, axis=1, keepdims=True)
    return float(np.sum((P @ V) * dO))


def _finite_diff(Q, K, V, B, dO, name, eps=1e-6):
    base = {"Q": Q, "K": K, "V": V}[name]
    pos = {"Q": 0, "K": 1, "V": 2}[name]
    out = np.zeros_like(base, dtype=np.float64)

    for idx in np.ndindex(base.shape):
        plus_args = [Q.copy(), K.copy(), V.copy()]
        minus_args = [Q.copy(), K.copy(), V.copy()]
        plus_args[pos][idx] += eps
        minus_args[pos][idx] -= eps
        plus = _attention_loss(*plus_args, B, dO)
        minus = _attention_loss(*minus_args, B, dO)
        out[idx] = (plus - minus) / (2 * eps)
    return out


def _stats(Q, K, B):
    d = Q.shape[1]
    S = Q @ K.T / np.sqrt(float(d)) + B
    m = np.max(S, axis=1)
    l = np.sum(np.exp(S - m[:, None]), axis=1)
    return m, l


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(11)
    Q = rng.normal(size=(3, 4)).astype(np.float64)
    K = rng.normal(size=(3, 4)).astype(np.float64)
    V = rng.normal(size=(3, 4)).astype(np.float64)
    B = rng.normal(size=(3, 3)).astype(np.float64) * 0.5
    dO = rng.normal(size=(3, 4)).astype(np.float64)
    m, l = _stats(Q, K, B)

    ref = (
        _finite_diff(Q, K, V, B, dO, "Q"),
        _finite_diff(Q, K, V, B, dO, "K"),
        _finite_diff(Q, K, V, B, dO, "V"),
    )

    try:
        got = sol.biased_flash_backward(Q, K, V, B, dO, m, l)
        err = max(
            float(np.max(np.abs(np.asarray(got[0], dtype=np.float64) - ref[0]))),
            float(np.max(np.abs(np.asarray(got[1], dtype=np.float64) - ref[1]))),
            float(np.max(np.abs(np.asarray(got[2], dtype=np.float64) - ref[2]))),
        )
    except Exception:
        err = float("inf")

    return {"max_abs_err": err}
