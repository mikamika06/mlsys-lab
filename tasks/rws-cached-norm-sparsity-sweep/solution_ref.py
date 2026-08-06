import math
import numpy as np


def wanda_masks_for_sparsities(W: np.ndarray, X: np.ndarray, sparsities: list[float]) -> list[np.ndarray]:
    """Wanda per-row unstructured pruning masks for several sparsities,
    reusing one activation-norm pass.

    Returns a list of (out_features, in_features) 0/1 masks, one per
    entry of `sparsities`, in the same order.
    """
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    n_samples, in_features = X.shape
    col_norm = []
    for j in range(in_features):
        acc = 0.0
        for i in range(n_samples):
            val = X[i, j]
            acc += val * val
        col_norm.append(math.sqrt(acc))

    out_features, _ = W.shape
    S = []
    for o in range(out_features):
        row = []
        for j in range(in_features):
            row.append(abs(W[o, j]) * col_norm[j])
        S.append(row)

    masks = []
    for s in sparsities:
        n_prune = int(round(s * in_features))
        mask_list = [[1] * in_features for _ in range(out_features)]
        for o in range(out_features):
            order = sorted(range(in_features), key=lambda j: S[o][j])
            prune_idx = order[:n_prune]
            for j in prune_idx:
                mask_list[o][j] = 0
        masks.append(np.array(mask_list, dtype=np.int64))
    return masks
