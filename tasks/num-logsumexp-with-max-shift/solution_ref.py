"""Reference solution for `num-logsumexp-with-max-shift`."""
from __future__ import annotations

import math


def logsumexp(x: list[list[float]], axis: int = -1) -> list[float]:
    """Numerically stable log(sum(exp(x))) along `axis`.

    Shifts by the per-slice max before exponentiating, so no term ever
    overflows: exp(x - m) <= 1 everywhere, and at least one term equals 1.
    """
    if not isinstance(x, list):
        val = float(x)
        return math.log(math.exp(val))

    if len(x) == 0:
        return []

    is_1d = all(not isinstance(item, list) for item in x)

    if is_1d:
        max_val = -float('inf')
        for val in x:
            v = float(val)
            if v > max_val:
                max_val = v
        acc = 0.0
        for val in x:
            acc += math.exp(float(val) - max_val)
        return math.log(acc) + max_val

    ndim = 2
    if axis < 0:
        axis += ndim

    if axis == 0:
        num_cols = len(x[0])
        result = []
        for j in range(num_cols):
            col = [row[j] for row in x]
            result.append(logsumexp(col, axis=0))
        return result
    elif axis == 1:
        return [logsumexp(row, axis=0) for row in x]
    else:
        raise ValueError(f"Axis {axis} out of bounds for ndim {ndim}")
