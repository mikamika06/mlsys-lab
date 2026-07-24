import numpy as np

def classify_mask_2_4(mask: np.ndarray):
    """
    Return a tuple (group_validity, overall) indicating whether each group of four
    consecutive elements along the last dimension sums to exactly two.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim == 0:
        raise ValueError("mask must have at least one dimension")
    if mask.shape[-1] % 4 != 0:
        raise ValueError("last dimension must be divisible by 4")
    groups = mask.reshape(*mask.shape[:-1], -1, 4)
    sums = np.sum(groups, axis=-1)
    group_validity = (sums == 2)
    overall = bool(np.all(group_validity))
    return group_validity, overall
