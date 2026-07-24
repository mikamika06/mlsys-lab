import numpy as np


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _attention(Q, K, V):
    d = Q.shape[1]
    scores = (Q @ K.T) / np.sqrt(d)
    return _softmax(scores) @ V


def _oracle(Q, K, V, B):
    full = _attention(Q, K, V)
    best_k = None
    best_err = None
    n = Q.shape[0]
    for k in range(1, B):
        w = B - k
        idx = np.concatenate(
            [
                np.arange(k),
                np.arange(n - w, n),
            ]
        )
        idx = np.unique(idx)
        approx = _attention(Q, K[idx], V[idx])
        err = float(np.sum((full - approx) ** 2))
        if best_err is None or err < best_err:
            best_err = err
            best_k = k
    return best_k


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(7)
    cases = []
    for n, d, B in [
        (8, 3, 4),
        (10, 4, 5),
        (12, 2, 6),
        (9, 5, 3),
    ]:
        Q = rng.normal(size=(n, d))
        K = rng.normal(size=(n, d))
        V = rng.normal(size=(n, d))
        cases.append((Q, K, V, B))

    ok = 1.0
    for Q, K, V, B in cases:
        try:
            got = sol.optimize_sink_window_split(Q, K, V, B)
        except Exception:
            ok = 0.0
            break
        ref = _oracle(Q, K, V, B)
        if int(got) != int(ref):
            ok = 0.0
            break
    return {"argmin_index": ok}
