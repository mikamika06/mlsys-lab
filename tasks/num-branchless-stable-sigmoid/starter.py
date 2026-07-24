import numpy as np


def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    """Return the logistic sigmoid of ``x`` without ever overflowing.

    x -- float64 array, |x| may be as large as 1e4
    returns -- float64 array of the same shape with values in [0, 1]
    """
    raise NotImplementedError('your code here')
