"""Reference solution for `num-logsumexp-with-max-shift`."""
from __future__ import annotations

import math
import numpy as np


def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log(sum(exp(x))) along `axis`.

    Shifts by the per-slice max before exponentiating, so no term ever
    overflows: exp(x - m) <= 1 everywhere, and at least one term equals 1.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        return math.log(math.exp(float(x)))

    if axis < 0:
        axis += x.ndim

    shape = x.shape
    reduced_shape = shape[:axis] + shape[axis + 1:]

    m_data = np.empty(reduced_shape, dtype=np.float64)
    s_data = np.empty(reduced_shape, dtype=np.float64)

    it = np.nditer(m_data, flags=['multi_index'], op_flags=['writeonly'])
    while not it.finished:
        idx = it.multi_index
        sub_indices = idx[:axis] + (slice(None),) + idx[axis:]
        slice_1d = x[sub_indices]

        max_val = -float('inf')
        for val in slice_1d:
            v = float(val)
            if v > max_val:
                max_val = v
        m_data[idx] = max_val

        acc = 0.0
        for val in slice_1d:
            acc += math.exp(float(val) - max_val)
        s_data[idx] = math.log(acc) + max_val

        it.iternext()

    return s_data
