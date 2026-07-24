import numpy as np

E2M1_MAX = 6.0


def _e4m3_nonneg_grid() -> np.ndarray:
    bias = 7
    vals = set()
    for e in range(16):
        for m in range(8):
            if e == 15 and m == 7:
                continue  # NaN
            if e == 0:
                v = (m / 8.0) * (2.0 ** (1 - bias))
            else:
                v = (1.0 + m / 8.0) * (2.0 ** (e - bias))
            vals.add(v)
    return np.array(sorted(vals), dtype=np.float64)


_GRID = _e4m3_nonneg_grid()


def _round_to_e4m3(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    idx = np.argmin(np.abs(x[..., None] - _GRID[None, :]), axis=-1)
    return _GRID[idx]


def nvfp4_block_scales(W: np.ndarray, group_size: int, per_tensor_scale: float) -> np.ndarray:
    """NVFP4 second-level block-scale factorization.

    For each contiguous block of `group_size` elements, the raw scale
    that lands the block's absmax exactly on E2M1's max representable
    value (6.0) is `max(|block|) / (6 * per_tensor_scale)`; that raw
    scale is then rounded to the nearest representable E4M3 magnitude.
    """
    W = np.asarray(W, dtype=np.float64)
    n_blocks = W.shape[0] // group_size
    blocks = W[:n_blocks * group_size].reshape(n_blocks, group_size)
    absmax = np.max(np.abs(blocks), axis=1)
    raw = absmax / (E2M1_MAX * per_tensor_scale)
    return _round_to_e4m3(raw)
