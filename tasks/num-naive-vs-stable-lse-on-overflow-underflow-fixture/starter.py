import numpy as np


def logsumexp_stable(x: np.ndarray) -> float:
    """
    Numerically stable log-sum-exp: ``log(sum(exp(x)))``.

    Must return a finite, accurate result even when ``x`` contains entries
    far outside the range where ``exp`` is representable in float64 (both
    large positive entries that would overflow ``exp``, and large negative
    entries that would underflow it to zero).
    """
    raise NotImplementedError('your code here')
