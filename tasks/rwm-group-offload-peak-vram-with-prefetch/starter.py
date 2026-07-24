import numpy as np


def offload_peak_vram(group_sizes: np.ndarray, leaf_sizes: np.ndarray) -> dict:
    """
    Return a dict with keys "group", "sequential", "model" giving the peak
    resident bytes under group offloading with prefetch, leaf-level
    sequential offloading, and no offload, respectively. See task.md for the
    exact formulas.
    """
    raise NotImplementedError('your code here')
