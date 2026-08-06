import numpy as np


def classify_scheduling(active: np.ndarray) -> str:
    """Classify a per-iteration active-ID trace as "static" or "continuous".

    active: (T, N) array, active[t, i] truthy iff sequence i is active at
    iteration t. Continuous iff some iteration admits a new ID while a
    previous member is still active; static otherwise.
    """
    active = np.asarray(active).astype(bool)
    T = active.shape[0]
    N = active.shape[1]
    for t in range(1, T):
        has_new = False
        has_continuing = False
        for i in range(N):
            prev_val = bool(active[t - 1, i])
            curr_val = bool(active[t, i])
            new_id = curr_val and (not prev_val)
            continuing = prev_val and curr_val
            if new_id:
                has_new = True
            if continuing:
                has_continuing = True
        if has_new and has_continuing:
            return "continuous"
    return "static"
