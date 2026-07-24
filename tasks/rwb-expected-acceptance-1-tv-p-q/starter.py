import numpy as np


def expected_acceptance(p: np.ndarray, q: np.ndarray) -> float:
    """Expected single-token speculative-decoding acceptance probability.

    p: target distribution, 1-D probability vector (sums to 1).
    q: draft distribution, same shape as p (sums to 1).

    Return sum_v min(p_v, q_v), equivalently 1 - TV(p, q) where TV is the
    total variation distance 0.5 * sum_v |p_v - q_v|.
    """
    raise NotImplementedError('your code here')
