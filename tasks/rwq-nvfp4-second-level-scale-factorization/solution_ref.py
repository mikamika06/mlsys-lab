import numpy as np

E2M1_MAX = 6.0


def _e4m3_nonneg_grid() -> np.ndarray:
    bias = 7
    vals = set()
    for e in range(16):
        for m in range(8):
            if e == 15 and m == 7:
                continue
            if e == 0:
                v = (m / 8.0) * (2.0 ** (1 - bias))
            else:
                v = (1.0 + m / 8.0) * (2.0 ** (e - bias))
            vals.add(v)
    return np.array(sorted(vals), dtype=np.float64)


_GRID = _e4m3_nonneg_grid()


def _round_to_e4m3(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    orig_shape = x.shape
    x_flat = x.ravel()
    out_list = []
    for val in x_flat:
        best_idx = 0
        min_diff = 0.0
        for i, g in enumerate(_GRID):
            diff = val - g
            if diff < 0.0:
                diff = -diff
            if i == 0 or diff < min_diff:
                min_diff = diff
                best_idx = i
        out_list.append(_GRID[best_idx])
    return np.array(out_list, dtype=np.float64).reshape(orig_shape)


def nvfp4_block_scales(W: np.ndarray, group_size: int, per_tensor_scale: float) -> np.ndarray:
    """NVFP4 second-level block-scale factorization.

    For each contiguous block of `group_size` elements, the raw scale
    that lands the block's absmax exactly on E2M1's max representable
    value (6.0) is `max(|block|) / (6 * per_tensor_scale)`; that raw
    scale is then rounded to the nearest representable E4M3 magnitude.
    """
    W = np.asarray(W, dtype=np.float64)
    n_blocks = W.shape[0] // group_size
    raw_list = []
    for i in range(n_blocks):
        block = W[i * group_size : (i + 1) * group_size]
        max_val = 0.0
        for j in range(group_size):
            v = block[j]
            if v < 0.0:
                v = -v
            if j == 0 or v > max_val:
                max_val = v
        raw = max_val / (E2M1_MAX * per_tensor_scale)
        raw_list.append(raw)
    raw_array = np.array(raw_list, dtype=np.float64)
    return _round_to_e4m3(raw_array)
