import numpy as np


def is_valid_l1_mask(w: np.ndarray, mask: np.ndarray, amount: float) -> bool:
    """
    True iff `mask` is exactly PyTorch's L1Unstructured pruning mask for
    `w` at sparsity `amount`: prune the k = round(amount * n) smallest-
    |w| entries (mask False there), keep the rest (mask True).
    """
    w = np.asarray(w, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    n = w.size

    k = int(round(amount * n))
    true_mask_list = [True] * n
    if k > 0:
        indices = sorted(range(n), key=lambda i: abs(w[i]))
        for i in indices[:k]:
            true_mask_list[i] = False

    if mask.size != n:
        return False

    for i in range(n):
        if bool(mask[i]) != bool(true_mask_list[i]):
            return False

    return True
