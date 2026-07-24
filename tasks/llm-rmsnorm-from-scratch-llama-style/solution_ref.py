import numpy as np

def rmsnorm(x, weight, eps=1e-6):
    x_arr = np.asarray(x, dtype=np.float64)
    w_arr = np.asarray(weight, dtype=x_arr.dtype)
    denom = np.sqrt(np.mean(x_arr**2) + eps)
    return w_arr * (x_arr / denom)
