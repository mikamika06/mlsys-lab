import numpy as np


def double_rounding_counterexample():
    """x = 1 + 2^-24 + 2^-54 rounds differently one step vs two.

    Direct to binary32: x is strictly above the midpoint 1 + 2^-24, so it
    rounds up to 1 + 2^-23.
    Via binary64: 2^-54 is a quarter of the binary64 ulp on [1,2), so x snaps
    down onto the exactly representable 1 + 2^-24; that is now an exact binary32
    tie, and ties-to-even picks 1.0 (even significand).
    """
    den = 1 << 54
    num = den + (1 << 30) + 1          # (1 + 2^-24 + 2^-54) * 2^54

    direct = np.float32(1.0) + np.float32(2.0 ** -23)
    doubled = np.float32(1.0)
    return num, den, direct, doubled
