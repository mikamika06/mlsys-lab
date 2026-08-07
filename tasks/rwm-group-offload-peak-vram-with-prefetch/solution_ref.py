def offload_peak_vram(group_sizes: list[float], leaf_sizes: list[float]) -> dict:
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
    max_group = group_sizes[0]
    for i in range(1, len(group_sizes)):
        val = group_sizes[i]
        if val > max_group:
            max_group = val

    max_leaf = leaf_sizes[0]
    for i in range(1, len(leaf_sizes)):
        val = leaf_sizes[i]
        if val > max_leaf:
            max_leaf = val

    sum_group = 0.0
    for i in range(len(group_sizes)):
        sum_group += group_sizes[i]

    return {
        "group": float(2.0 * max_group),
        "sequential": float(max_leaf),
        "model": float(sum_group),
    }
