import numpy as np


def double_rounding_counterexample():
    """Return (num, den, direct, doubled).

    num/den -- exact rational x with den a power of two and 1 <= x < 2
    direct  -- x rounded straight to binary32 (ties to even)
    doubled -- x rounded to binary64 first, then narrowed to binary32
    The two must differ.
    """
    raise NotImplementedError('your code here')
