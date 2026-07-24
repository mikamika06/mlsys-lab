import numpy as np


def classify_scheduling(active: np.ndarray) -> str:
    """Classify a per-iteration active-ID trace as "static" or "continuous".

    active: (T, N) array, active[t, i] truthy iff sequence i is active at
    iteration t. Continuous iff some iteration admits a new ID while a
    previous member is still active; static otherwise.
    """
    active = np.asarray(active).astype(bool)
    T = active.shape[0]
    for t in range(1, T):
        prev = active[t - 1]
        curr = active[t]
        new_ids = curr & ~prev
        continuing = prev & curr
        if new_ids.any() and continuing.any():
            return "continuous"
    return "static"
