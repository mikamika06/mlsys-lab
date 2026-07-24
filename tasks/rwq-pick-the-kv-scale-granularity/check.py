import numpy as np


def _quantize(x, kind):
    h, t, d = x.shape
    if kind == 0:
        scale = np.max(np.abs(x)) / 127.0
        scales = np.array([scale])
        return np.clip(np.round(x / scale), -127, 127) * scale, scales
    if kind == 1:
        scale = np.max(np.abs(x), axis=(0, 2), keepdims=True) / 127.0
        return np.clip(np.round(x / scale), -127, 127) * scale, scale.reshape(-1)
    scale = np.max(np.abs(x), axis=(1, 2), keepdims=True) / 127.0
    return np.clip(np.round(x / scale), -127, 127) * scale, scale.reshape(-1)


def _attention(Q, K, V):
    scores = np.matmul(Q, np.transpose(K, (0, 2, 1))) / np.sqrt(K.shape[-1])
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= np.sum(probs, axis=-1, keepdims=True)
    return np.matmul(probs, V)


def _oracle(K, V, Q, budget):
    ref = _attention(Q, K, V)
    lam = 0.001
    best = None
    best_i = -1
    for i in range(3):
        Kq, ks = _quantize(K, i)
        Vq, vs = _quantize(V, i)
        out = _attention(Q, Kq, Vq)
        mse = float(np.mean((ref - out) ** 2))
        scale_bytes = (ks.size + vs.size) * 4
        cost = mse + lam * max(0, scale_bytes - budget)
        if best is None or cost < best:
            best = cost
            best_i = i
    return best_i


def grade(sol, fx) -> dict:
    cases = []
    rng = np.random.default_rng(42)
    for h, t, d, budget in [
        (1, 8, 16, 16),
        (2, 7, 8, 32),
        (4, 12, 4, 64),
        (3, 5, 16, 8),
    ]:
        K = rng.normal(0, 2, size=(h, t, d)).astype(np.float64)
        V = rng.normal(0, 1, size=(h, t, d)).astype(np.float64)
        Q = rng.normal(0, 1, size=(h, 1, d)).astype(np.float64)
        cases.append((K, V, Q, budget))

    ok = 1.0
    for K, V, Q, budget in cases:
        try:
            got = sol.choose_kv_scale_granularity(K, V, Q, budget)
            ref = _oracle(K, V, Q, budget)
            if int(got) != ref:
                ok = 0.0
                break
        except Exception:
            ok = 0.0
            break
    return {"argmin_index": ok}
