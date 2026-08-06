import math
import numpy as np

def predict_rounding_results(arr: np.ndarray, mode: str) -> np.ndarray:
    """Return the float32 array that would result from converting each element
    of ``arr`` using the specified rounding mode.

    Parameters
    ----------
    arr : np.ndarray
        One‑dimensional array of dtype ``float64``.
    mode : str
        Either ``"nearest"`` (round‑to‑nearest‑even) or ``"trunc"``
        (round toward zero).

    Returns
    -------
    np.ndarray
        Array of the same shape, dtype ``float32``.
    """
    if mode == "nearest":
        n = len(arr)
        res = np.zeros(n, dtype=np.float32)
        for i in range(n):
            res[i] = float(arr[i])
        return res
    elif mode == "trunc":
        n = len(arr)
        res = np.zeros(n, dtype=np.float32)
        for i in range(n):
            val = float(arr[i])
            res[i] = math.copysign(float(math.trunc(val)), val)
        return res
    else:
        raise ValueError("mode must be 'nearest' or 'trunc'")
