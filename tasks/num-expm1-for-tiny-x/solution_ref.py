import numpy as np


def exp_minus_one(x):
    """Accurate e**x - 1, including for |x| far below the fp64 epsilon."""
    x = np.asarray(x, dtype=np.float64)
    u = np.exp(x)
    d = u - 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
        lu = np.log(u)
        val = d * (x / lu)

    out = np.where(d == 0.0, x, val)          # u rounded to 1 -> e^x - 1 == x
    out = np.where(u == 0.0, -1.0, out)       # exp underflowed -> -1
    return out
