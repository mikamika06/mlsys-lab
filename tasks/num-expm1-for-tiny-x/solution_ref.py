import math
import numpy as np


def exp_minus_one(x):
    """Accurate e**x - 1, including for |x| far below the fp64 epsilon."""
    x_arr = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x_arr, dtype=np.float64)

    for i in range(x_arr.size):
        xi = float(x_arr.flat[i])
        try:
            u = math.exp(xi)
        except OverflowError:
            u = float("inf")

        d = u - 1.0

        if u == 0.0:
            val = -1.0
        elif d == 0.0:
            val = xi
        else:
            try:
                lu = math.log(u)
            except ValueError:
                lu = float("nan")
            val = d * (xi / lu)

        out.flat[i] = val

    return out
