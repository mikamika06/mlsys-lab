import numpy as np

E4M3_MAX = 448.0


def _e4m3_grid_pos() -> np.ndarray:
    """All non-negative finite E4M3 (4 exponent bits, 3 mantissa bits, bias 7) values."""
    vals = set()
    for exp in range(16):
        for mant in range(8):
            if exp == 15 and mant == 7:
                continue
            if exp == 0:
                v = (2.0 ** -6) * (mant / 8.0)
            else:
                v = (2.0 ** (exp - 7)) * (1.0 + mant / 8.0)
            vals.add(v)
    return np.array(sorted(vals), dtype=np.float64)


_GRID = _e4m3_grid_pos()


def _cast_e4m3(x: np.ndarray) -> np.ndarray:
    """Round each element to the nearest representable E4M3 value (clamped to +-448)."""
    x = np.asarray(x, dtype=np.float64)

    def cast_scalar(val):
        sign = 1.0 if val > 0 else (-1.0 if val < 0 else 0.0)
        absx = min(max(abs(val), 0.0), E4M3_MAX)
        low = 0
        high = len(_GRID)
        while low < high:
            mid = (low + high) // 2
            if _GRID[mid] < absx:
                low = mid + 1
            else:
                high = mid
        idx = low
        idx = max(1, min(idx, len(_GRID) - 1))
        lo = _GRID[idx - 1]
        hi = _GRID[idx]
        snapped = hi if (hi - absx) < (absx - lo) else lo
        return sign * snapped

    def process_sub(arr):
        if arr.ndim == 0:
            return cast_scalar(float(arr))
        return [process_sub(arr[i]) for i in range(arr.shape[0])]

    nested = process_sub(x)
    return np.array(nested, dtype=np.float64)


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

    M = W64.shape[0]
    K = W64.shape[1]
    N = X64.shape[1]

    amax_w = 0.0
    for i in range(M):
        for k in range(K):
            val = abs(W64[i, k])
            if val > amax_w:
                amax_w = val
    scale_w = amax_w / E4M3_MAX if amax_w > 0 else 1.0

    amax_x = []
    for j in range(N):
        max_col = 0.0
        for k in range(K):
            val = abs(X64[k, j])
            if val > max_col:
                max_col = val
        amax_x.append(max_col)

    scale_x = [(val / E4M3_MAX if val > 0 else 1.0) for val in amax_x]

    Wq_list = []
    for i in range(M):
        row = []
        for k in range(K):
            row.append(W64[i, k] / scale_w)
        Wq_list.append(row)
    Wq = _cast_e4m3(np.array(Wq_list, dtype=np.float64))

    Xq_list = []
    for k in range(K):
        row = []
        for j in range(N):
            row.append(X64[k, j] / scale_x[j])
        Xq_list.append(row)
    Xq = _cast_e4m3(np.array(Xq_list, dtype=np.float64))

    Y_list = []
    for i in range(M):
        row = []
        for j in range(N):
            acc = 0.0
            for k in range(K):
                acc += Wq[i, k] * Xq[k, j]
            row.append(acc * scale_w * scale_x[j])
        Y_list.append(row)

    Y = np.array(Y_list, dtype=np.float64)
    return Y.astype(np.float32)
