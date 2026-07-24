import numpy as np

NF4_LEVELS = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
], dtype=np.float64)

BLOCK_SIZE = 64


def _oracle(w: np.ndarray, block_size: int):
    n = w.shape[0]
    nb = n // block_size
    wb = w.reshape(nb, block_size)
    absmax = np.max(np.abs(wb), axis=1)
    absmax_safe = np.where(absmax == 0, 1.0, absmax)
    normalized = wb / absmax_safe[:, None]
    diffs = np.abs(normalized[:, :, None] - NF4_LEVELS[None, None, :])
    idx = np.argmin(diffs, axis=-1).astype(np.int64)
    deq = (NF4_LEVELS[idx] * absmax_safe[:, None]).reshape(n)
    return idx.reshape(n), deq


def _fail():
    return {"index_exact_match": 0.0, "max_abs_err": float("inf")}


def grade(sol, fx) -> dict:
    w = fx["nf4_w"]
    idx_ref, deq_ref = _oracle(w, BLOCK_SIZE)

    try:
        out = sol.nf4_quantize_dequantize(w.copy(), BLOCK_SIZE)
    except Exception:
        return _fail()

    try:
        idx_got, deq_got = out
        idx_got = np.asarray(idx_got).astype(np.int64).reshape(-1)
        deq_got = np.asarray(deq_got, dtype=np.float64).reshape(-1)
    except Exception:
        return _fail()

    if idx_got.shape != idx_ref.shape or deq_got.shape != deq_ref.shape:
        return _fail()

    idx_match = float(np.array_equal(idx_got, idx_ref))
    err = float(np.max(np.abs(deq_got - deq_ref)))
    return {"index_exact_match": idx_match, "max_abs_err": err}
