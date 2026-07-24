import numpy as np

def _oracle(w, block_size=64):
    """NumPy reference: reshape into blocks and take per-block max|w|."""
    w = np.asarray(w, dtype=np.float64)
    n = len(w)
    pad_len = (-n) % block_size
    if pad_len:
        w = np.concatenate([w, np.zeros(pad_len)])
    return np.max(np.abs(w.reshape(-1, block_size)), axis=1)

def grade(sol, fx) -> dict:
    rng = np.random.RandomState(0)
    cases = [
        (rng.randn(256).astype(np.float32), 64),
        (rng.randn(128).astype(np.float64), 64),
        (rng.randn(64).astype(np.float32), 64),
        (rng.randn(512).astype(np.float64), 64),
        (rng.randn(100).astype(np.float64), 64),   # not a multiple of 64
        (rng.randn(200).astype(np.int32).astype(np.float64), 64),
    ]
    worst = 0.0
    for w, bs in cases:
        oracle = _oracle(w, bs)
        try:
            got = np.asarray(sol.blockwise_absmax(w, bs), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}
        if got.shape != oracle.shape:
            return {"max_abs_err": float("inf")}
        err = float(np.max(np.abs(got - oracle)))
        worst = max(worst, err)
    return {"max_abs_err": worst}
