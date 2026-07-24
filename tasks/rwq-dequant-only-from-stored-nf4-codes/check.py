import numpy as np

_NF4_LEVELS = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
], dtype=np.float64)


def _oracle(idx, absmax, block_size):
    idx = np.asarray(idx, dtype=np.int64)
    absmax = np.asarray(absmax, dtype=np.float64)
    block = idx.size // block_size
    codes = _NF4_LEVELS[idx]
    scales = np.repeat(absmax[:block], block_size)
    return codes * scales


def grade(sol, fx) -> dict:
    idx = fx["nf4_idx"]
    absmax = fx["nf4_absmax"]
    block_size = 64

    ref = _oracle(idx, absmax, block_size)

    try:
        got = sol.nf4_dequantize(idx.copy(), absmax.copy(), block_size)
        got = np.asarray(got, dtype=np.float64)
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}
        err = float(np.max(np.abs(got - ref)))
    except Exception:
        return {"max_abs_err": float("inf")}

    return {"max_abs_err": err}
