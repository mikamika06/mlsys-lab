import itertools
import math
import numpy as np


def stable_log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable log‑softmax.

    Parameters
    ----------
    logits : np.ndarray
        Input array of arbitrary shape.
    axis : int, default -1
        Axis along which to apply the softmax.

    Returns
    -------
    np.ndarray
        Array of same shape and dtype float64 containing the log‑softmax values.
    """
    logits = np.asarray(logits, dtype=np.float64)
    out = np.empty(logits.shape, dtype=np.float64)

    ndim = logits.ndim
    if axis < 0:
        axis += ndim

    dim_len = logits.shape[axis]
    ranges = [range(logits.shape[d]) for d in range(ndim) if d != axis]

    for other in itertools.product(*ranges):
        first_idx = list(other)
        first_idx.insert(axis, 0)
        max_val = float(logits[tuple(first_idx)])

        for i in range(1, dim_len):
            idx = list(other)
            idx.insert(axis, i)
            val = float(logits[tuple(idx)])
            if val > max_val:
                max_val = val

        sum_exp = 0.0
        for i in range(dim_len):
            idx = list(other)
            idx.insert(axis, i)
            val = float(logits[tuple(idx)])
            sum_exp += math.exp(val - max_val)

        log_sum_exp = math.log(sum_exp)

        for i in range(dim_len):
            idx = list(other)
            idx.insert(axis, i)
            val = float(logits[tuple(idx)])
            out[tuple(idx)] = val - max_val - log_sum_exp

    return out
