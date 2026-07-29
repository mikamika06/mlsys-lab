import math

import numpy as np


def _power_of_2_slopes(n):
    start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3.0)))
    ratio = start
    return [start * (ratio ** i) for i in range(n)]


def _slopes_list(n_heads):
    if math.log2(n_heads).is_integer():
        return _power_of_2_slopes(n_heads)
    closest = 2 ** math.floor(math.log2(n_heads))
    extra = _slopes_list(2 * closest)[0::2][: n_heads - closest]
    return _power_of_2_slopes(closest) + extra


def alibi_slopes(n_heads):
    if n_heads <= 0:
        raise ValueError("n_heads must be positive")
    return np.array(_slopes_list(n_heads), dtype=np.float64)
