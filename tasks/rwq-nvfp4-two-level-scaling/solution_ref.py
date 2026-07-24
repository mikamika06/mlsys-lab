import numpy as np

FP4_MAX = 6.0
FP8_MAX = 448.0

E2M1_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)


def _e4m3_nonneg_grid() -> np.ndarray:
    vals = set()
    for exp in range(16):
        for mant in range(8):
            if exp == 15 and mant == 7:
                continue
            if exp == 0:
                val = (2.0 ** -6) * (mant / 8.0)
            else:
                val = (2.0 ** (exp - 7)) * (1.0 + mant / 8.0)
            vals.add(val)
    return np.array(sorted(vals), dtype=np.float64)


E4M3_GRID = _e4m3_nonneg_grid()


def _snap(vals: np.ndarray, grid: np.ndarray) -> np.ndarray:
    vals = np.atleast_1d(np.asarray(vals, dtype=np.float64))
    diffs = np.abs(vals[:, None] - grid[None, :])
    idx = np.argmin(diffs, axis=1)
    return grid[idx]


def nvfp4_two_level_quantize(w: np.ndarray, block_size: int = 16):
    w = np.asarray(w, dtype=np.float64)
    n = w.shape[0]
    nb = n // block_size
    wb = w.reshape(nb, block_size)

    tensor_amax = float(np.max(np.abs(w)))
    global_scale = tensor_amax / (FP4_MAX * FP8_MAX)

    block_amax = np.max(np.abs(wb), axis=1)
    block_scale_fp32 = block_amax / FP4_MAX
    block_scale_scaled = np.where(block_amax == 0, 0.0, block_scale_fp32 / global_scale)
    block_scale_scaled = np.clip(block_scale_scaled, 0.0, FP8_MAX)
    block_scales_e4m3 = _snap(block_scale_scaled, E4M3_GRID)

    eff_scale = block_scales_e4m3 * global_scale
    eff_scale_safe = np.where(eff_scale == 0, 1.0, eff_scale)

    normalized = wb / eff_scale_safe[:, None]
    sign = np.sign(normalized)
    mag = np.clip(np.abs(normalized), 0.0, FP4_MAX)
    mag_snapped = _snap(mag.reshape(-1), E2M1_MAG).reshape(mag.shape)
    codes = sign * mag_snapped

    dequant = (codes * eff_scale[:, None]).reshape(n)
    return global_scale, block_scales_e4m3, codes.reshape(n), dequant
