import numpy as np


def repeat_kv(x, n_rep):
    x = np.asarray(x)
    if n_rep == 1:
        return x
    return np.repeat(x, n_rep, axis=-3)
