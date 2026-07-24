import numpy as np

def hamming_and_jaccard(B):
    B = np.asarray(B, dtype=np.uint8)
    B_bool = B.astype(bool)
    H = np.sum(B_bool[:, None] != B_bool[None, :], axis=2).astype(np.int64)
    inter = (B_bool[:, None] & B_bool[None, :]).sum(axis=2)
    union = (B_bool[:, None] | B_bool[None, :]).sum(axis=2)
    J = np.empty_like(inter, dtype=np.float64)
    mask = union == 0
    if np.any(mask):
        J[mask] = 1.0
    J[~mask] = inter[~mask] / union[~mask].astype(np.float64)
    return H, J
