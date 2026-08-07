import numpy as np

def score_magnitude(w):
    return np.abs(w)

def score_wanda(w, x):
    x_norm = np.linalg.norm(x, axis=1)
    return np.abs(w) * x_norm[None, :]
