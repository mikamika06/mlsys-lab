import numpy as np

def simulate_reordered_sum(matrix):
    flat = matrix.flatten()
    s1 = float(np.sum(flat))
    idx = np.argsort(np.abs(flat))
    s2 = float(np.sum(flat[idx]))
    return {"standard": s1, "reordered": s2, "delta": abs(s1 - s2)}
