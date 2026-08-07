def h2o_heavy_hitters(attn_matrix, capacity):
    import numpy as np
    arr = np.array(attn_matrix, dtype=float)
    if arr.size == 0 or capacity <= 0:
        return []
    scores = arr.sum(axis=0)
    if len(scores) <= capacity:
        return sorted(list(range(len(scores))))
    indices = np.argsort(scores)[-capacity:]
    return sorted(indices.tolist())
