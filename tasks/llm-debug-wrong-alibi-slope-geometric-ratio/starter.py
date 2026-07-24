import numpy as np


def alibi_slopes(n: int) -> np.ndarray:
    # TODO: wrong geometric ratio. This uses the head count as a divisor of
    # 2^-8 instead of making the exponent depend on each head index.
    h = np.arange(n, dtype=np.float64)
    ratio = (2.0 ** -8) / n
    return np.power(ratio, h).astype(np.float64)
