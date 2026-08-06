import numpy as np


def rtn_group_quantize(W: np.ndarray, group_size: int):
    """
    Per-row, per-group symmetric int4 round-to-nearest quantization, no
    error feedback. `d_in` is guaranteed divisible by `group_size`.

    For each row and each contiguous block of `group_size` columns:
      amax  = max(|W[row, block]|)
      scale = amax / 7   (or 1.0 if amax == 0)
      code  = clip(round(w / scale), -7, 7)
    Dequantized reconstruction is code * scale.

    Returns (codes, Wq):
      codes -- integer array, same shape as W, values in [-7, 7].
      Wq    -- float array, same shape as W, the dequantized reconstruction.
    """
    W = np.asarray(W, dtype=np.float64)
    d_out, d_in = W.shape
    n_groups = d_in // group_size

    codes = np.empty((d_out, d_in), dtype=np.int64)
    Wq = np.empty((d_out, d_in), dtype=np.float64)

    for g in range(n_groups):
        col_start = g * group_size
        col_end = col_start + group_size
        for r in range(d_out):
            amax = 0.0
            for c in range(col_start, col_end):
                val = W[r, c]
                abs_val = val if val >= 0.0 else -val
                if abs_val > amax:
                    amax = abs_val

            if amax > 0.0:
                scale = amax / 7.0
            else:
                scale = 1.0

            for c in range(col_start, col_end):
                val = W[r, c]
                scaled = val / scale
                rounded = round(scaled)

                if rounded > 7:
                    clipped = 7
                elif rounded < -7:
                    clipped = -7
                else:
                    clipped = int(rounded)

                codes[r, c] = clipped
                Wq[r, c] = float(clipped) * scale

    return codes, Wq
