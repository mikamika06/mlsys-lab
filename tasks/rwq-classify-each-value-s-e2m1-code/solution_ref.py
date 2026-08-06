import numpy as np

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
    out = np.zeros(x.shape, dtype=np.int64)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx_tuple = it.multi_index
        val = x[idx_tuple]
        absval = -val if val < 0 else val
        min_dist = float("inf")
        best_idx = 0
        for i in range(8):
            g = GRID[i]
            diff = absval - g
            dist = -diff if diff < 0 else diff
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        sign_bit = 1 if val < 0 else 0
        out[idx_tuple] = sign_bit * 8 + best_idx
        it.iternext()
    return out
