import numpy as np

def fp8_representability(values: np.ndarray):
    """
    Broken implementation that uses incorrect bias values and ignores subnormals.
    This will fail the exact_match gate on many test cases.
    """
    values = np.asarray(values, dtype=np.float64)
    abs_vals = np.abs(values)
    finite_mask = np.isfinite(values)

    # e4m3 with wrong bias (6 instead of 7) and max exponent field including all ones
    m1, e1 = 3, 4
    bias1 = (1 << (e1 - 1)) - 2          # WRONG: should be 7
    max_exp_field1 = (1 << e1) - 1       # WRONG: includes 1111 as normal
    max_exponent1 = max_exp_field1 - bias1
    min_normal1 = 2 ** (1 - bias1)
    max_finite1 = (2 - 2 ** (-m1)) * 2 ** max_exponent1

    mask_e4m3 = finite_mask & (
        (abs_vals == 0) |
        ((abs_vals >= min_normal1) & (abs_vals <= max_finite1))
    )

    # e5m2 ignoring subnormals entirely
    m2, e2 = 2, 5
    bias2 = (1 << (e2 - 1)) - 1          # correct bias
    max_exp_field2 = (1 << e2) - 2       # correct
    max_exponent2 = max_exp_field2 - bias2
    min_normal2 = 2 ** (1 - bias2)
    max_finite2 = (2 - 2 ** (-m2)) *
