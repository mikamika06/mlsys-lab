import numpy as np


def _oracle(tokens, q, k, v, S, W):
    T = len(tokens)
    sink = np.arange(min(S, T), dtype=np.int64)
    start = max(0, T - W)
    window = np.arange(start, T, dtype=np.int64)
    idx = np.unique(np.concatenate([sink, window]))

    pos = np.empty(len(idx), dtype=np.int64)
    for j, i in enumerate(idx):
        if i < S:
            pos[j] = i
        else:
            pos[j] = S + i - start

    kk = k[idx]
    vv = v[idx]
    logits = q @ kk.T / np.sqrt(k.shape[1])
    logits = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    out = weights @ vv
    return idx, pos, out


def grade(sol, fx) -> dict:
    cases = [
        (
            np.arange(10),
            np.array([[1.0, 0.5], [0.2, -0.4]]),
            np.arange(20, dtype=np.float64).reshape(10, 2) / 10,
            np.arange(30, dtype=np.float64).reshape(10, 3) / 7,
            2,
            4,
        ),
        (
            np.arange(6),
            np.array([[0.1, 0.9, -0.2]]),
            np.arange(18, dtype=np.float64).reshape(6, 3) / 5,
            np.arange(12, dtype=np.float64).reshape(6, 2) / 11,
            1,
            3,
        ),
    ]

    ok = 1.0
    for args in cases:
        ref = _oracle(*args)
        try:
            got = sol.streaming_attention(*args)
            gi = np.asarray(got[0], dtype=np.int64)
            gp = np.asarray(got[1], dtype=np.int64)
            go = np.asarray(got[2], dtype=np.float64)
        except Exception:
            ok = 0.0
            break

        if not np.array_equal(gi, ref[0]):
            ok = 0.0
            break
        if not np.array_equal(gp, ref[1]):
            ok = 0.0
            break
        if not np.allclose(go, ref[2], rtol=1e-10, atol=1e-10):
            ok = 0.0
            break

    return {"exact_match": ok}
