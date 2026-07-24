import numpy as np


def offload_peak_vram(group_sizes: np.ndarray, leaf_sizes: np.ndarray) -> dict:
    """
    Peak resident bytes under three offload strategies for the same model:

    - "group": group offloading with prefetch. While one group executes, the
      next is already resident (prefetched), so up to two groups (the
      largest one, twice) are resident at the worst moment.
    - "sequential": leaf-level offloading, no prefetch overlap -> only the
      single largest leaf is ever resident.
    - "model": no offload -> the whole model (sum of all group sizes) is
      resident.
    """
    group_sizes = np.asarray(group_sizes, dtype=np.float64)
    leaf_sizes = np.asarray(leaf_sizes, dtype=np.float64)
    return {
        "group": float(2.0 * np.max(group_sizes)),
        "sequential": float(np.max(leaf_sizes)),
        "model": float(np.sum(group_sizes)),
    }
