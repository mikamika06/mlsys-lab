import numpy as np

from mlsys import scorers


def _quant_dequant_1d(x: np.ndarray, bits: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    qmax = (1 << bits) - 1
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax <= xmin:
        return x.copy()
    scale = (xmax - xmin) / qmax
    zero = round(-xmin / scale)
    zero = min(max(zero, 0), qmax)
    codes = np.clip(np.round(x / scale + zero), 0, qmax)
    return (codes - zero) * scale


def _grouped_dequant(W: np.ndarray, bits: int, group_size) -> np.ndarray:
    W = np.asarray(W, dtype=np.float64)
    rows, cols = W.shape
    if group_size is None:
        flat = W.reshape(-1)
        return _quant_dequant_1d(flat, bits).reshape(rows, cols)

    out = np.empty_like(W)
    for r in range(rows):
        row = W[r]
        for start in range(0, cols, group_size):
            seg = row[start:start + group_size]
            out[r, start:start + group_size] = _quant_dequant_1d(seg, bits)
    return out


def _ref_frontier(W, bit_options, group_size_options) -> np.ndarray:
    W = np.asarray(W, dtype=np.float64)
    mse = np.zeros((len(bit_options), len(group_size_options)), dtype=np.float64)
    for bi, bits in enumerate(bit_options):
        for gi, g in enumerate(group_size_options):
            W_hat = _grouped_dequant(W, bits, g)
            mse[bi, gi] = float(np.mean((W_hat - W) ** 2))
    return mse


def _build_W() -> np.ndarray:
    rng = np.random.default_rng(0)
    rows, cols = 12, 256
    W = rng.normal(loc=0.0, scale=1.0, size=(rows, cols))
    flat = W.reshape(-1)
    idx = rng.choice(flat.size, size=8, replace=False)
    flat[idx] += rng.normal(loc=0.0, scale=8.0, size=8)
    return flat.reshape(rows, cols)


def grade(sol, fx) -> dict:
    W = _build_W()
    bit_options = [2, 4]
    group_size_options = [None, 128, 64, 32]

    ref = _ref_frontier(W, bit_options, group_size_options)

    try:
        got = sol.bitwidth_group_mse_frontier(W.copy(), list(bit_options), list(group_size_options))
    except Exception:
        return {"mse": float("inf"), "monotone": 0.0}

    try:
        got = np.asarray(got, dtype=np.float64)
    except Exception:
        return {"mse": float("inf"), "monotone": 0.0}

    if got.shape != ref.shape:
        return {"mse": float("inf"), "monotone": 0.0}

    mse_err = scorers.rel_err(ref, got)

    mono_ok = 1.0
    for bi in range(got.shape[0]):
        row = got[bi]
        if np.any(np.diff(row) > 1e-9):
            mono_ok = 0.0
            break

    return {"mse": mse_err, "monotone": mono_ok}
