import numpy as np


def quantify_non_associativity(arr):
    flat = arr.astype(np.float32).flatten()
    n = len(flat)
    if n == 0:
        return 0.0
    s1 = float(np.sum(flat))
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    s2 = float(np.sum(flat[perm]))
    inv_perm = np.argsort(perm)
    s3 = float(np.sum(flat[inv_perm]))
    vals = np.array([s1, s2, s3], dtype=np.float32)
    mean_val = np.mean(vals)
    mse = float(np.mean((vals - mean_val) ** 2))
    return mse
