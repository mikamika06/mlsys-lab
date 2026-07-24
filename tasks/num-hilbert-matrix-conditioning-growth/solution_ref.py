import numpy as np


def hilbert_condition_numbers(ns):
    result = []
    for n in ns:
        idx = np.arange(n, dtype=np.float64)
        H = 1.0 / (idx[:, None] + idx[None, :] + 1.0)
        s = np.linalg.svd(H, compute_uv=False)
        result.append(np.log10(s[0] / s[-1]))
    return np.asarray(result, dtype=np.float64)
