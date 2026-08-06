import math
import numpy as np

def stable_sigmoid(z: np.ndarray) -> np.ndarray:
    z_arr = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z_arr, dtype=np.float64)
    flat_z = z_arr.flat
    flat_out = out.flat
    for i in range(z_arr.size):
        val = flat_z[i]
        if val >= 0.0:
            flat_out[i] = 1.0 / (1.0 + math.exp(-val))
        else:
            exp_val = math.exp(val)
            flat_out[i] = exp_val / (1.0 + exp_val)
    return out
