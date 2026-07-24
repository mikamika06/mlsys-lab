import numpy as np


def load_balancing_aux_loss(router_probs: np.ndarray) -> float:
    probs = np.asarray(router_probs, dtype=np.float64)
    n, e = probs.shape
    assignments = np.argmax(probs, axis=1)
    counts = np.bincount(assignments, minlength=e)
    f = counts.astype(np.float64) / n
    p = np.mean(probs, axis=0)
    return float(e * np.sum(f * p))
