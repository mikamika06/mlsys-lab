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
    vals_arr = np.atleast_1d(np.asarray(vals, dtype=np.float64))
    orig_shape = vals_arr.shape
    flat_vals = vals_arr.ravel()
    res = []
    for v in flat_vals:
        best_g = grid[0]
        min_diff = abs(v - best_g)
        for g in grid:
            diff = abs(v - g)
            if diff < min_diff:
                min_diff = diff
                best_g = g
        res.append(best_g)
    return np.array(res, dtype=np.float64).reshape(orig_shape)


def nvfp4_two_level_quantize(w: np.ndarray, block_size: int = 16):
    w = np.asarray(w, dtype=np.float64)
    n = w.shape[0]
    nb = n // block_size
    wb = w.reshape(nb, block_size)

    tensor_amax = 0.0
    for x in w:
        val = abs(x)
        if val > tensor_amax:
            tensor_amax = val
    tensor_amax = float(tensor_amax)
    global_scale = tensor_amax / (FP4_MAX * FP8_MAX)

    block_amax_list = []
    for i in range(nb):
        m = 0.0
        for j in range(block_size):
            val = abs(wb[i, j])
            if val > m:
                m = val
        block_amax_list.append(m)
    block_amax = np.array(block_amax_list, dtype=np.float64)

    block_scale_fp32_list = []
    for b_amax in block_amax:
        block_scale_fp32_list.append(b_amax / FP4_MAX)
    block_scale_fp32 = np.array(block_scale_fp32_list, dtype=np.float64)

    block_scale_scaled_list = []
    for b_amax, bs_fp32 in zip(block_amax, block_scale_fp32):
        if b_amax == 0.0:
            block_scale_scaled_list.append(0.0)
        else:
            block_scale_scaled_list.append(bs_fp32 / global_scale)
    block_scale_scaled = np.array(block_scale_scaled_list, dtype=np.float64)

    block_scale_clipped_list = []
    for val in block_scale_scaled:
        if val < 0.0:
            block_scale_clipped_list.append(0.0)
        elif val > FP8_MAX:
            block_scale_clipped_list.append(FP8_MAX)
        else:
            block_scale_clipped_list.append(val)
    block_scale_scaled = np.array(block_scale_clipped_list, dtype=np.float64)

    block_scales_e4m3 = _snap(block_scale_scaled, E4M3_GRID)

    eff_scale_list = []
    for bs in block_scales_e4m3:
        eff_scale_list.append(bs * global_scale)
    eff_scale = np.array(eff_scale_list, dtype=np.float64)

    eff_scale_safe_list = []
    for es in eff_scale:
        if es == 0.0:
            eff_scale_safe_list.append(1.0)
        else:
            eff_scale_safe_list.append(es)
    eff_scale_safe = np.array(eff_scale_safe_list, dtype=np.float64)

    normalized_list = []
    for i in range(nb):
        row = []
        es_safe = eff_scale_safe[i]
        for j in range(block_size):
            row.append(wb[i, j] / es_safe)
        normalized_list.append(row)
    normalized = np.array(normalized_list, dtype=np.float64)

    sign_list = []
    for i in range(nb):
        row = []
        for j in range(block_size):
            val = normalized[i, j]
            if val > 0.0:
                row.append(1.0)
            elif val < 0.0:
                row.append(-1.0)
            else:
                row.append(val)
        sign_list.append(row)
    sign = np.array(sign_list, dtype=np.float64)

    mag_list = []
    for i in range(nb):
        row = []
        for j in range(block_size):
            m = abs(normalized[i, j])
            if m < 0.0:
                m = 0.0
            elif m > FP4_MAX:
                m = FP4_MAX
            row.append(m)
        mag_list.append(row)
    mag = np.array(mag_list, dtype=np.float64)

    mag_snapped = _snap(mag.reshape(-1), E2M1_MAG).reshape(mag.shape)

    codes_list = []
    for i in range(nb):
        row = []
        for j in range(block_size):
            row.append(sign[i, j] * mag_snapped[i, j])
        codes_list.append(row)
    codes = np.array(codes_list, dtype=np.float64)

    dequant_list = []
    for i in range(nb):
        es = eff_scale[i]
        for j in range(block_size):
            dequant_list.append(codes[i, j] * es)
    dequant = np.array(dequant_list, dtype=np.float64)

    return global_scale, block_scales_e4m3, codes.reshape(n), dequant
