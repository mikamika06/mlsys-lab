import numpy as np


def q4_1_quantize(w):
    """Port of ggml's quantize_row_q4_1_reference.

    w: array (N_b, 32) of raw block weights.
    Returns (d, m, codes):
      d, m: (N_b,) fp16-rounded per-block scale/min.
      codes: (N_b, 32) uint8 codes in [0, 15], computed with full-precision id.
    """
    w = np.asarray(w, dtype=np.float64)
    mn = w.min(axis=1)
    mx = w.max(axis=1)
    d = (mx - mn) / 15.0
    inv_d = np.where(d != 0, 1.0 / np.where(d != 0, d, 1.0), 0.0)

    x0 = (w - mn[:, None]) * inv_d[:, None]
    codes = np.minimum(15, np.floor(x0 + 0.5)).astype(np.uint8)
    codes = np.clip(codes, 0, 15)

    d16 = np.float16(d).astype(np.float64)
    m16 = np.float16(mn).astype(np.float64)
    return d16, m16, codes
