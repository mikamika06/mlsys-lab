def int4_groupwise_quant(
    W: list[list[float]], group_size: int = 32
) -> tuple[list[list[int]], list[list[float]]]:
    """
    W: (rows, cols) weight matrix; `cols` must be a multiple of
        `group_size`.

    Symmetric int4 quantization, applied independently per row and per
    contiguous group of `group_size` values along the columns:

        amax  = max(abs(group))
        scale = amax / 8              (1.0 if amax == 0, to avoid /0)
        code  = clip(round(group / scale), -8, 7)

    Returns (codes, scales):
      codes: (rows, cols) int array, values in [-8, 7].
      scales: (rows, cols // group_size) float array, one scale per row
        per group.
    """
    rows = len(W)
    cols = len(W[0]) if rows > 0 else 0
    n_groups = cols // group_size

    scales_list = []
    codes_list = []

    for r in range(rows):
        row_scales = []
        row_codes = []
        for g in range(n_groups):
            start = g * group_size
            end = start + group_size

            max_val = 0.0
            for i in range(start, end):
                val = W[r][i]
                abs_val = val if val >= 0 else -val
                if abs_val > max_val:
                    max_val = abs_val

            if max_val == 0.0:
                scale = 1.0
            else:
                scale = max_val / 8.0
            row_scales.append(scale)

            for i in range(start, end):
                val = W[r][i]
                divided = val / scale
                rounded = round(divided)
                clipped = -8 if rounded < -8 else (7 if rounded > 7 else rounded)
                row_codes.append(int(clipped))

        scales_list.append(row_scales)
        codes_list.append(row_codes)

    return codes_list, scales_list
