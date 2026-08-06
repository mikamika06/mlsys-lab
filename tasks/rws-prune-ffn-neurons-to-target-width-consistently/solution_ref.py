import numpy as np

def prune_ffn_neurons(
    up_proj: np.ndarray,
    down_proj: np.ndarray,
    target_width: int
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """
    Return the indices of the most important neurons and the corresponding sliced
    projection matrices.

    Neuron importance is defined as the sum of absolute weights in both
    projections.  The top `target_width` neurons are kept.
    """
    hidden_dim = up_proj.shape[0]
    out_dim = up_proj.shape[1]
    in_dim = down_proj.shape[0]

    importance = []
    for i in range(hidden_dim):
        s = 0.0
        for j in range(out_dim):
            val = up_proj[i, j]
            if val < 0:
                s -= val
            else:
                s += val
        for k in range(in_dim):
            val = down_proj[k, i]
            if val < 0:
                s -= val
            else:
                s += val
        importance.append(s)

    indexed_importance = []
    for i in range(hidden_dim):
        indexed_importance.append((-importance[i], i))

    for i in range(1, hidden_dim):
        key = indexed_importance[i]
        j = i - 1
        while j >= 0 and indexed_importance[j] > key:
            indexed_importance[j + 1] = indexed_importance[j]
            j -= 1
        indexed_importance[j + 1] = key

    idx = []
    for i in range(target_width):
        idx.append(indexed_importance[i][1])

    idx_sorted = list(idx)
    for i in range(1, target_width):
        key = idx_sorted[i]
        j = i - 1
        while j >= 0 and idx_sorted[j] > key:
            idx_sorted[j + 1] = idx_sorted[j]
            j -= 1
        idx_sorted[j + 1] = key

    up_sliced = np.empty((target_width, out_dim), dtype=up_proj.dtype)
    for i in range(target_width):
        src_row = idx_sorted[i]
        for j in range(out_dim):
            up_sliced[i, j] = up_proj[src_row, j]

    down_sliced = np.empty((in_dim, target_width), dtype=down_proj.dtype)
    for i in range(in_dim):
        for j in range(target_width):
            src_col = idx_sorted[j]
            down_sliced[i, j] = down_proj[i, src_col]

    return list(idx_sorted), up_sliced, down_sliced
