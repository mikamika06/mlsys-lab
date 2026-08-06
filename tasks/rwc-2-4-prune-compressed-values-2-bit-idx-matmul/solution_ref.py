import numpy as np


def prune24_compress_and_matmul(W: np.ndarray, X: np.ndarray):
    """
    2:4 structured sparsity (the NVIDIA Ampere sparse-tensor-core format):
    every group of 4 consecutive columns in a row keeps only its 2
    largest-magnitude values; the other 2 are pruned to zero.

    W : (m, n) weight matrix, n divisible by 4.
    X : (n, p) input/activation matrix.

    Build the COMPRESSED representation of the pruned W:
      values  : (m, n//2) float64 -- the 2 kept values per group of 4, in
                their ORIGINAL left-to-right column order within the group.
      indices : (m, n//2) uint8   -- each kept value's position (0-3)
                within its group of 4 (same order as `values`). Ties in
                |value| are broken in favor of the LOWER index.

    Reconstruct the pruned matrix from (values, indices) -- scatter each
    kept value back to `group_start + index` -- and compute the matmul
    against the (unpruned) X.

    Returns (mask, values, indices, output):
      mask   : (m, n) int array, 1 where a value survived pruning, else 0.
      values, indices : as described above.
      output : (m, p) = (pruned W) @ X.
    """
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    m, n = W.shape
    groups = n // 4
    p = X.shape[1]

    mask = np.zeros((m, n), dtype=np.int64)
    values = np.zeros((m, groups * 2), dtype=np.float64)
    indices = np.zeros((m, groups * 2), dtype=np.uint8)

    for i in range(m):
        for g in range(groups):
            grp = W[i, g * 4:(g + 1) * 4]
            
            abs_vals = [abs(grp[k]) for k in range(4)]
            order = []
            for k in range(4):
                best_idx = -1
                best_val = -1.0
                for idx in range(4):
                    if idx not in order:
                        val = abs_vals[idx]
                        if best_idx == -1 or val > best_val:
                            best_val = val
                            best_idx = idx
                order.append(best_idx)
            
            kept_unsorted = order[:2]
            keep = []
            for k in range(4):
                if k in kept_unsorted:
                    keep.append(k)
            
            for idx in keep:
                mask[i, g * 4 + idx] = 1
            
            for s_idx, idx in enumerate(keep):
                values[i, g * 2 + s_idx] = grp[idx]
                indices[i, g * 2 + s_idx] = np.uint8(idx)

    pruned = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for g in range(groups):
            for slot in range(2):
                col = g * 4 + int(indices[i, g * 2 + slot])
                pruned[i, col] = values[i, g * 2 + slot]

    output = np.zeros((m, p), dtype=np.float64)
    for i in range(m):
        for j in range(p):
            acc = 0.0
            for k in range(n):
                acc += pruned[i, k] * X[k, j]
            output[i, j] = acc

    return mask, values, indices, output
