import numpy as np


def classify_scheduling(active: np.ndarray) -> str:
    """Classify a per-iteration active-ID trace as "static" or "continuous".

    active: (T, N) array, active[t, i] truthy iff sequence i is active at
    iteration t. Returns the string "static" or "continuous".
    """
    raise NotImplementedError('your code here')
