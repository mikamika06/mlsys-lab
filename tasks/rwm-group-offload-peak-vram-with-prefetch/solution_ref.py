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

    max_group = group_sizes[0]
    for i in range(1, group_sizes.shape[0]):
        val = group_sizes[i]
        if val > max_group:
            max_group = val

    max_leaf = leaf_sizes[0]
    for i in range(1, leaf_sizes.shape[0]):
        val = leaf_sizes[i]
        if val > max_leaf:
            max_leaf = val

    sum_group = 0.0
    for i in range(group_sizes.shape[0]):
        sum_group += group_sizes[i]

    return {
        "group": float(2.0 * max_group),
        "sequential": float(max_leaf),
        "model": float(sum_group),
    }
