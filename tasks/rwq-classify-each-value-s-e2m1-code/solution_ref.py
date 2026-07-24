import numpy as np

# OCP MX FP4 (E2M1) magnitude grid: 1 sign bit, 2 exponent bits, 1 mantissa bit.
GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def e2m1_classify(x: np.ndarray) -> np.ndarray:
    """Classify each element of `x` to its signed 4-bit E2M1 code.

    The code is `sign_bit * 8 + magnitude_index`, where `magnitude_index` is
    the index (0..7) of the nearest value in `GRID` to `abs(x)`
    (round-to-nearest, ties toward the smaller-magnitude code -- exactly
    what `np.argmin` returns on the first minimal distance), and `sign_bit`
    is `1` if `x < 0` else `0`.

    Parameters
    ----------
    x : np.ndarray
        Any shape, real-valued.

    Returns
    -------
    codes : np.ndarray, same shape as `x`, dtype int64, values in [0, 15].
    """
    x = np.asarray(x, dtype=np.float64)
    absx = np.abs(x)
    dist = np.abs(absx[..., None] - GRID)
    idx = np.argmin(dist, axis=-1)
    sign_bit = (x < 0).astype(np.int64)
    return (sign_bit * 8 + idx).astype(np.int64)
