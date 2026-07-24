import numpy as np

E4M3_MAX = 448.0


def _e4m3_grid() -> np.ndarray:
    bias = 7
    vals = set()
    for sign in (1.0, -1.0):
        for e in range(16):
            for m in range(8):
                if e == 15 and m == 7:
                    continue  # NaN
                if e == 0:
                    v = (m / 8.0) * (2.0 ** (1 - bias))
                else:
                    v = (1.0 + m / 8.0) * (2.0 ** (e - bias))
                vals.add(sign * v)
    return np.array(sorted(vals), dtype=np.float64)


_GRID = _e4m3_grid()


def _round_to_e4m3(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    flat = x.reshape(-1)
    idx = np.argmin(np.abs(flat[:, None] - _GRID[None, :]), axis=1)
    return _GRID[idx].reshape(x.shape)


def qfloat8_weight_quant(W: np.ndarray):
    """Per-tensor scale (to E4M3's max magnitude 448) + nearest-E4M3 cast.

    Returns (scale, e4m3_values, W_hat).
    """
    W = np.asarray(W, dtype=np.float64)
    amax = float(np.max(np.abs(W)))
    scale = amax / E4M3_MAX if amax > 0 else 1.0
    W_scaled = W / scale
    codes = _round_to_e4m3(W_scaled)
    W_hat = codes * scale
    return scale, codes, W_hat
