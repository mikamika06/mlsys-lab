import math
import numpy as np


def fp8_representability(values: np.ndarray):
    """
    Return two boolean masks indicating whether each element of *values* can be
    represented exactly in the e4m3 and e5m2 FP8 formats.
    """
    values = np.asarray(values, dtype=np.float64)

    m1, e1 = 3, 4
    bias1 = (1 << (e1 - 1)) - 1
    max_exp_field1 = (1 << e1) - 2
    max_exponent1 = max_exp_field1 - bias1
    min_subnormal1 = 2 ** (1 - bias1 - m1)
    max_finite1 = (2 - 2 ** (-m1)) * 2 ** max_exponent1

    m2, e2 = 2, 5
    bias2 = (1 << (e2 - 1)) - 1
    max_exp_field2 = (1 << e2) - 2
    max_exponent2 = max_exp_field2 - bias2
    min_subnormal2 = 2 ** (1 - bias2 - m2)
    max_finite2 = (2 - 2 ** (-m2)) * 2 ** max_exponent2

    flat_vals = values.reshape(-1)
    n = len(flat_vals)

    mask_e4m3_flat = np.empty(n, dtype=bool)
    mask_e5m2_flat = np.empty(n, dtype=bool)

    for i in range(n):
        v = float(flat_vals[i])
        is_finite = math.isfinite(v)

        if v < 0:
            abs_v = -v
        else:
            abs_v = v

        is_zero = abs_v == 0.0

        cond1 = is_zero or (
            abs_v >= min_subnormal1 and abs_v <= max_finite1
        )
        mask_e4m3_flat[i] = is_finite and cond1

        cond2 = is_zero or (
            abs_v >= min_subnormal2 and abs_v <= max_finite2
        )
        mask_e5m2_flat[i] = is_finite and cond2

    mask_e4m3 = mask_e4m3_flat.reshape(values.shape)
    mask_e5m2 = mask_e5m2_flat.reshape(values.shape)

    return mask_e4m3, mask_e5m2
