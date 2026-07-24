import numpy as np


def wanda_masks_for_sparsities(W: np.ndarray, X: np.ndarray, sparsities: list[float]) -> list[np.ndarray]:
    """Wanda per-row unstructured pruning masks for several sparsities,
    reusing one activation-norm pass.

    Returns a list of (out_features, in_features) 0/1 masks, one per
    entry of `sparsities`, in the same order.
    """
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    col_norm = np.linalg.norm(X, axis=0)          # computed once
    S = np.abs(W) * col_norm[None, :]              # reused for every sparsity

    out_features, in_features = W.shape
    masks = []
    for s in sparsities:
        n_prune = int(round(s * in_features))
        mask = np.ones_like(W, dtype=np.int64)
        for o in range(out_features):
            order = np.argsort(S[o], kind="stable")
            prune_idx = order[:n_prune]
            mask[o, prune_idx] = 0
        masks.append(mask)
    return masks
