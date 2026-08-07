def rtn_group_quantize(
    W: list[list[float]], group_size: int
) -> tuple[list[list[int]], list[list[float]]]:
    """
    Per-row, per-group symmetric int4 round-to-nearest quantization, no
    error feedback. `d_in` is guaranteed divisible by `group_size`.

    For each row and each contiguous block of `group_size` columns:
      amax  = max(|W[row, block]|)
      scale = amax / 7   (or 1.0 if amax == 0)
      code  = clip(round(w / scale), -7, 7)
    Dequantized reconstruction is code * scale.

    Returns (codes, Wq):
      codes -- integer list of lists, same shape as W, values in [-7, 7].
      Wq    -- float list of lists, same shape as W, the dequantized reconstruction.
    """
    d_out = len(W)
    if d_out == 0:
        return [], []
    d_in = len(W[0])
    n_groups = d_in // group_size

    codes = [[0] * d_in for _ in range(d_out)]
    Wq = [[0.0] * d_in for _ in range(d_out)]

    for g in range(n_groups):
        col_start = g * group_size
        col_end = col_start + group_size
        for r in range(d_out):
            amax = 0.0
            for c in range(col_start, col_end):
                val = W[r][c]
                abs_val = val if val >= 0.0 else -val
                if abs_val > amax:
                    amax = abs_val

            if amax > 0.0:
                scale = amax / 7.0
            else:
                scale = 1.0

            for c in range(col_start, col_end):
                val = W[r][c]
                scaled = val / scale
                rounded = round(scaled)

                if rounded > 7:
                    clipped = 7
                elif rounded < -7:
                    clipped = -7
                else:
                    clipped = int(rounded)

                codes[r][c] = clipped
                Wq[r][c] = float(clipped) * scale

    return codes, Wq
