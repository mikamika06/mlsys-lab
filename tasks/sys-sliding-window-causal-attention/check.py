import numpy as np


def _full_mask(n, window):
    rows = np.arange(n).reshape(-1, 1)
    cols = np.arange(n).reshape(1, -1)
    return (cols <= rows) & (rows - cols < window)


def _reference(Q, K, V, mask):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    n, d = Q.shape
    scores = (Q @ K.T) / np.sqrt(d)
    masked = np.where(mask, scores, -np.inf)
    masked = masked - np.max(masked, axis=-1, keepdims=True)
    e = np.exp(masked)
    p = e / np.sum(e, axis=-1, keepdims=True)
    return p @ V


def _cases(fx):
    rng = np.random.default_rng(0)
    out = []

    # The fixture case: n=12, window=4 (mask.npy shape must match).
    n = fx["mask"].shape[0]
    Q = rng.standard_normal((n, 4))
    K = rng.standard_normal((n, 4))
    V = rng.standard_normal((n, 4))
    out.append((Q, K, V, 4, 5, fx["mask"]))

    # Additional inline-mask cases: (n, d, window, block_size).
    specs = [
        (1, 3, 1, 1),
        (6, 3, 1, 2),      # window=1 -> self only
        (7, 5, 3, 3),
        (8, 4, 8, 3),      # window == n -> full causal
        (9, 6, 20, 4),     # window > n -> full causal
        (10, 4, 4, 10),    # block_size == n -> single tile
        (11, 4, 4, 1),     # block_size == 1 -> row-by-row
    ]
    for n2, d2, w2, bs2 in specs:
        Q2 = rng.standard_normal((n2, d2))
        K2 = rng.standard_normal((n2, d2))
        V2 = rng.standard_normal((n2, d2))
        out.append((Q2, K2, V2, w2, bs2, _full_mask(n2, w2)))

    return out


def grade(sol, fx) -> dict:
    worst = 0.0
    for Q, K, V, window, block_size, mask in _cases(fx):
        ref = _reference(Q, K, V, mask)
        try:
            got = np.asarray(
                sol.sliding_window_attention_tiled(Q.copy(), K.copy(), V.copy(), window, block_size),
                dtype=np.float64,
            )
        except Exception:
            return {"max_abs_err": float("inf")}

        if got.shape != ref.shape or not np.all(np.isfinite(got)):
            return {"max_abs_err": float("inf")}

        worst = max(worst, float(np.max(np.abs(got - ref))))

    return {"max_abs_err": worst}
