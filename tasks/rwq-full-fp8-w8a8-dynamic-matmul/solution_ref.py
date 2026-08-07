E4M3_MAX = 448.0


def _e4m3_grid_pos() -> list[float]:
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
    return sorted(list(vals))


_GRID = _e4m3_grid_pos()


def _cast_e4m3(x: list[list[float]]) -> list[list[float]]:
    """Round each element to the nearest representable E4M3 value (clamped to +-448)."""
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
        if not isinstance(arr, list):
            return cast_scalar(float(arr))
        return [process_sub(item) for item in arr]

    return process_sub(x)


def fp8_dynamic_matmul(W: list[list[float]], X: list[list[float]]) -> list[list[float]]:
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
    M = len(W)
    K = len(W[0])
    N = len(X[0])

    amax_w = 0.0
    for i in range(M):
        for k in range(K):
            val = abs(W[i][k])
            if val > amax_w:
                amax_w = val
    scale_w = amax_w / E4M3_MAX if amax_w > 0 else 1.0

    amax_x = []
    for j in range(N):
        max_col = 0.0
        for k in range(K):
            val = abs(X[k][j])
            if val > max_col:
                max_col = val
        amax_x.append(max_col)

    scale_x = [(val / E4M3_MAX if val > 0 else 1.0) for val in amax_x]

    Wq_list = []
    for i in range(M):
        row = []
        for k in range(K):
            row.append(W[i][k] / scale_w)
        Wq_list.append(row)
    Wq = _cast_e4m3(Wq_list)

    Xq_list = []
    for k in range(K):
        row = []
        for j in range(N):
            row.append(X[k][j] / scale_x[j])
        Xq_list.append(row)
    Xq = _cast_e4m3(Xq_list)

    Y_list = []
    for i in range(M):
        row = []
        for j in range(N):
            acc = 0.0
            for k in range(K):
                acc += Wq[i][k] * Xq[k][j]
            row.append(acc * scale_w * scale_x[j])
        Y_list.append(row)

    return Y_list
