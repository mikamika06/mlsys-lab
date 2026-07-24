import numpy as np


def smoothquant_migrate(W, X, s):
    scale = np.asarray(s, dtype=np.float64).reshape(1, -1)
    return np.asarray(W, dtype=np.float64) * scale, np.asarray(X, dtype=np.float64) / scale
