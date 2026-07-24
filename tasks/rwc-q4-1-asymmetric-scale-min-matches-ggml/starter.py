import numpy as np


def q4_1_quantize(w):
    """Port of ggml's quantize_row_q4_1_reference (asymmetric Q4_1).

    w: array (N_b, 32) of raw block weights.

    For each block: min/max over the 32 values, d = (max-min)/15,
    id = 1/d (0 if d==0), code = min(15, floor((w-min)*id + 0.5))
    computed at full precision, THEN d and min are rounded to float16.

    Returns (d, m, codes):
      d, m: (N_b,) fp16-rounded per-block scale/min (as float64/float32).
      codes: (N_b, 32) uint8 codes in [0, 15].
    """
    raise NotImplementedError('your code here')
