import numpy as np

E4M3_MAX = 448.0


def _e4m3_grid_pos() -> np.ndarray:
    """All non-negative finite E4M3 (4 exponent bits, 3 mantissa bits, bias 7) values."""
    vals = set()
    for exp in range(16):
        for mant in range(8):
            if exp == 15 and mant == 7:
                continue  # NaN code point
            if exp == 0:
                v = (2.0 ** -6) * (mant / 8.0)  # subnormal
            else:
                v = (2.0 ** (exp - 7)) * (1.0 + mant / 8.0)  # normal
            vals.add(v)
    return np.array(sorted(vals), dtype=np.float64)


_GRID = _e4m3_grid_pos()


def _cast_e4m3(x: np.ndarray) -> np.ndarray:
    """Round each element to the nearest representable E4M3 value (clamped to +-448)."""
    x = np.asarray(x, dtype=np.float64)
    sign = np.sign(x)
    absx = np.clip(np.abs(x), 0.0, E4M3_MAX)
    idx = np.searchsorted(_GRID, absx)
    idx = np.clip(idx, 1, len(_GRID) - 1)
    lo = _GRID[idx - 1]
    hi = _GRID[idx]
    snapped = np.where((hi - absx) < (absx - lo), hi, lo)
    return sign * snapped


def fp8_dynamic_matmul(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    FP8 E4M3 W8A8 "dynamic" quantized matmul: Y ~= W @ X.

    - `W` (M, K): weights, quantized with a single per-tensor scale.
    - `X` (K, N): activations, quantized with a per-token scale (one scale
      per column, i.e. per token) computed on the fly from `X` itself
      (that's the "dynamic" part -- no calibration pass).

    Both operands are cast to the nearest representable E4M3 value before
    the matmul, then the integer-like matmul result is dequantized by
    multiplying back by scale_w * scale_x[token].
    """
    W64 = np.asarray(W, dtype=np.float64)
    X64 = np.asarray(X, dtype=np.float64)

    amax_w = float(np.max(np.abs(W64)))
    scale_w = amax_w / E4M3_MAX if amax_w > 0 else 1.0

    amax_x = np.max(np.abs(X64), axis=0)  # per-token (per-column of X)
    scale_x = np.where(amax_x > 0, amax_x / E4M3_MAX, 1.0)

    Wq = _cast_e4m3(W64 / scale_w)
    Xq = _cast_e4m3(X64 / scale_x[None, :])

    Y = (Wq @ Xq) * scale_w * scale_x[None, :]
    return Y.astype(np.float32)
