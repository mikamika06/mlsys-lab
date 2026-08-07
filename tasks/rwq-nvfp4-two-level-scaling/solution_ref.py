FP4_MAX = 6.0
FP8_MAX = 448.0

E2M1_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _e4m3_nonneg_grid() -> list[float]:
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
    return sorted(list(vals))


E4M3_GRID = _e4m3_nonneg_grid()


def _snap(vals: list[float], grid: list[float]) -> list[float]:
    res = []
    for v in vals:
        best_g = grid[0]
        min_diff = abs(v - best_g)
        for g in grid:
            diff = abs(v - g)
            if diff < min_diff:
                min_diff = diff
                best_g = g
        res.append(float(best_g))
    return res


def nvfp4_two_level_quantize(w: list[float], block_size: int = 16) -> tuple[float, list[float], list[float], list[float]]:
    n = len(w)
    nb = n // block_size
    wb = [w[i * block_size:(i + 1) * block_size] for i in range(nb)]

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
            val = abs(wb[i][j])
            if val > m:
                m = val
        block_amax_list.append(m)
    block_amax = block_amax_list

    block_scale_fp32_list = []
    for b_amax in block_amax:
        block_scale_fp32_list.append(b_amax / FP4_MAX)
    block_scale_fp32 = block_scale_fp32_list

    block_scale_scaled_list = []
    for b_amax, bs_fp32 in zip(block_amax, block_scale_fp32):
        if b_amax == 0.0:
            block_scale_scaled_list.append(0.0)
        else:
            block_scale_scaled_list.append(bs_fp32 / global_scale)
    block_scale_scaled = block_scale_scaled_list

    block_scale_clipped_list = []
    for val in block_scale_scaled:
        if val < 0.0:
            block_scale_clipped_list.append(0.0)
        elif val > FP8_MAX:
            block_scale_clipped_list.append(FP8_MAX)
        else:
            block_scale_clipped_list.append(val)
    block_scale_scaled = block_scale_clipped_list

    block_scales_e4m3 = _snap(block_scale_scaled, E4M3_GRID)

    eff_scale_list = []
    for bs in block_scales_e4m3:
        eff_scale_list.append(bs * global_scale)
    eff_scale = eff_scale_list

    eff_scale_safe_list = []
    for es in eff_scale:
        if es == 0.0:
            eff_scale_safe_list.append(1.0)
        else:
            eff_scale_safe_list.append(es)
    eff_scale_safe = eff_scale_safe_list

    normalized_list = []
    for i in range(nb):
        row = []
        es_safe = eff_scale_safe[i]
        for j in range(block_size):
            row.append(wb[i][j] / es_safe)
        normalized_list.append(row)
    normalized = normalized_list

    sign_list = []
    for i in range(nb):
        row = []
        for j in range(block_size):
            val = normalized[i][j]
            if val > 0.0:
                row.append(1.0)
            elif val < 0.0:
                row.append(-1.0)
            else:
                row.append(val)
        sign_list.append(row)
    sign = sign_list

    mag_list = []
    for i in range(nb):
        row = []
        for j in range(block_size):
            m = abs(normalized[i][j])
            if m < 0.0:
                m = 0.0
            elif m > FP4_MAX:
                m = FP4_MAX
            row.append(m)
        mag_list.append(row)
    mag = mag_list

    mag_flat = []
    for row in mag:
        for val in row:
            mag_flat.append(val)

    mag_snapped_flat = _snap(mag_flat, E2M1_MAG)

    mag_snapped = []
    for i in range(nb):
        row = mag_snapped_flat[i * block_size : (i + 1) * block_size]
        mag_snapped.append(row)

    codes_list = []
    for i in range(nb):
        row = []
        for j in range(block_size):
            row.append(sign[i][j] * mag_snapped[i][j])
        codes_list.append(row)
    codes = codes_list

    dequant_list = []
    for i in range(nb):
        es = eff_scale[i]
        for j in range(block_size):
            dequant_list.append(codes[i][j] * es)
    dequant = dequant_list

    codes_flat = []
    for row in codes:
        for val in row:
            codes_flat.append(val)

    return global_scale, block_scales_e4m3, codes_flat, dequant
