from __future__ import annotations

import numpy as np


def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log(sum(exp(x))) along `axis`.

    Parameters
    ----------
    x : np.ndarray
        Input array, values may range anywhere from -1e4 to 1e4.
    axis : int
        Axis to reduce over (like `np.sum`'s `axis`, with `keepdims=False`).

    Returns
    -------
    np.ndarray
        `log(sum(exp(x), axis=axis))`, computed WITHOUT ever overflowing:
        shift by the per-slice max before exponentiating.
    """
    raise NotImplementedError('your code here')
