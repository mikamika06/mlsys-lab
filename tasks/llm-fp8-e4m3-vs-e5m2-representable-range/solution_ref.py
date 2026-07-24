import numpy as np

def fp8_representability(values: np.ndarray):
    """
    Return two boolean masks indicating whether each element of *values* can be
    represented exactly in the e4m3 and e5m2 FP8 formats.
    """
    values = np.asarray(values, dtype=np.float64)
    abs_vals = np.abs(values)
    finite_mask = np.isfinite(values)

    # --- e4m3 -------------------------------------------------------------
    m1, e1 = 3, 4
    bias1 = (1 << (e1 - 1)) - 1          # 7
    max_exp_field1 = (1 << e1) - 2       # 14
    max_exponent1 = max_exp_field1 - bias1
    min_normal1 = 2 ** (1 - bias1)
    min_subnormal1 = 2 ** (1 - bias1 - m1)
    max_finite1 = (2 - 2 ** (-m1)) * 2 ** max_exponent1

    mask_e4m3 = finite_mask & (
        (abs_vals == 0) |
        ((abs_vals >= min_subnormal1) & (abs_vals <= max_finite1))
    )

    # --- e5m2 -------------------------------------------------------------
    m2, e2 = 2, 5
    bias2 = (1 << (e2 - 1)) - 1          # 15
    max_exp_field2 = (1 << e2) - 2       # 30
    max_exponent2 = max_exp_field2 - bias2
    min_normal2 = 2 ** (1 - bias2)
    min_subnormal2 = 2 ** (1 - bias2 - m2)
    max_finite2 = (2 - 2 ** (-m2)) * 2 ** max_exponent2

    mask_e5m2 = finite_mask & (
        (abs_vals == 0) |
        ((abs_vals >= min_subnormal2) & (abs_vals <= max_finite2))
    )

    return mask_e4m3, mask_e5m2
