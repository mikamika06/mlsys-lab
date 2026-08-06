import math
import numpy as np


def alibi_slopes(n_heads: int) -> np.ndarray:
    """Return the ALiBi slopes for `n_heads` attention heads."""
    if not isinstance(n_heads, int) or n_heads <= 0:
        raise ValueError("n_heads must be a positive integer")

    def _get_power_of_two_slopes(k: int):
        start = 2 ** (-(k + 3))
        ratio = 2.0
        res = []
        for i in range(k):
            res.append(start * (ratio ** i))
        return np.array(res, dtype=np.float32)

    if n_heads & (n_heads - 1) == 0:
        return _get_power_of_two_slopes(n_heads)
    else:
        k = 2 ** int(math.floor(math.log2(n_heads)))
        slopes = _get_power_of_two_slopes(k).tolist()
        extra = alibi_slopes(n_heads - k).tolist()
        for item in extra:
            slopes.append(item)
        return np.array(slopes, dtype=np.float32)
