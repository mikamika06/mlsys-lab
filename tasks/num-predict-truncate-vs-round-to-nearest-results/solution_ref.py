import math

def predict_rounding_results(arr: list[float], mode: str) -> list[float]:
    """Return the list of floats that would result from converting each element
    of ``arr`` using the specified rounding mode.

    Parameters
    ----------
    arr : list[float]
        List of float values.
    mode : str
        Either ``"nearest"`` (round‑to‑nearest‑even) or ``"trunc"``
        (round toward zero).

    Returns
    -------
    list[float]
        List of the same length containing the converted values.
    """
    if mode == "nearest":
        res = []
        for val in arr:
            res.append(float(val))
        return res
    elif mode == "trunc":
        res = []
        for val in arr:
            val_f = float(val)
            res.append(math.copysign(float(math.trunc(val_f)), val_f))
        return res
    else:
        raise ValueError("mode must be 'nearest' or 'trunc'")
