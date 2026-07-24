"""Reference solution for `num-logsumexp-with-max-shift`."""
from __future__ import annotations

import numpy as np


def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log(sum(exp(x))) along `axis`.

    Shifts by the per-slice max before exponentiating, so no term ever
    overflows: exp(x - m) <= 1 everywhere, and at least one term equals 1.
    """
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    s = np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)) + m
    return np.squeeze(s, axis=axis)
