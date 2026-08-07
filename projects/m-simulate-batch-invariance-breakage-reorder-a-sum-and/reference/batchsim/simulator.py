import numpy as np

def simulate_reordering(logits, permutation):
    flat = logits.astype(np.float64)
    reordered = flat[permutation]
    delta = float(np.abs(np.sum(flat) - np.sum(reordered)))
    return reordered.astype(np.float32), delta
