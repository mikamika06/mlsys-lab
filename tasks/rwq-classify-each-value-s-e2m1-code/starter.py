import numpy as np


def e2m1_classify(x: np.ndarray) -> np.ndarray:
    """Classify each element of `x` to its signed 4-bit E2M1 code.

    The code is `sign_bit * 8 + magnitude_index`, where `magnitude_index` is
    the index (0..7) of the nearest value in the E2M1 magnitude grid
    `[0, 0.5, 1, 1.5, 2, 3, 4, 6]` to `abs(x)` (round-to-nearest, ties toward
    the smaller-magnitude code), and `sign_bit` is `1` if `x < 0` else `0`.

    Parameters
    ----------
    x : np.ndarray
        Any shape, real-valued.

    Returns
    -------
    codes : np.ndarray, same shape as `x`, dtype int64, values in [0, 15].
    """
    raise NotImplementedError('your code here')
