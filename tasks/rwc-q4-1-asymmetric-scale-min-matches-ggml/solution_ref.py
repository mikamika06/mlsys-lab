import math
import numpy as np


def q4_1_quantize(w):
    """Port of ggml's quantize_row_q4_1_reference.

    w: array (N_b, 32) of raw block weights.
    Returns (d, m, codes):
      d, m: (N_b,) fp16-rounded per-block scale/min.
      codes: (N_b, 32) uint8 codes in [0, 15], computed with full-precision id.
    """
    w = np.asarray(w, dtype=np.float64)
    n_b, n_cols = w.shape

    d16_list = []
    m16_list = []
    codes_list = []

    for i in range(n_b):
        row = w[i]
        mn = row[0]
        mx = row[0]
        for val in row[1:]:
            if val < mn:
                mn = val
            if val > mx:
                mx = val

        d = (mx - mn) / 15.0

        if d != 0:
            inv_d = 1.0 / d
        else:
            inv_d = 0.0

        row_codes = []
        for j in range(n_cols):
            x0 = (row[j] - mn) * inv_d
            val = math.floor(x0 + 0.5)
            if val > 15.0:
                val = 15.0
            if val < 0.0:
                val = 0.0
            row_codes.append(int(val))

        codes_list.append(row_codes)

        d16 = float(np.float16(d))
        m16 = float(np.float16(mn))
        d16_list.append(d16)
        m16_list.append(m16)

    d16_arr = np.array(d16_list, dtype=np.float64)
    m16_arr = np.array(m16_list, dtype=np.float64)
    codes_arr = np.array(codes_list, dtype=np.uint8)

    return d16_arr, m16_arr, codes_arr
