import numpy as np

def residual_norm_growth(block_fn, x):
    y = x
    for _ in range(12):
        y = block_fn(y)
    return float(np.linalg.norm(y) / np.linalg.norm(x))
