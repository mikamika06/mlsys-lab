"""Reference solution for `num-prove-kahan-sum-is-a-real-per-element-loop`."""
from __future__ import annotations

import numpy as np


def kahan_sum(x: np.ndarray) -> float:
    """Kahan (compensated) summation of `x`, via an explicit per-element
    Python loop. Tracks the low-order bits lost to rounding in a
    compensation term `c` and folds them back in on the next addition,
    so the running error does not accumulate with `len(x)`.
    """
    s = 0.0
    c = 0.0
    for v in x:
        y = float(v) - c
        t = s + y
        c = (t - s) - y
        s = t
    return s
