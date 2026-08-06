import numpy as np


def classify_mask(mask, L, H, d_ff, target_heads, target_ff):
    arr = np.asarray(mask, dtype=bool)
    if arr.shape != (L, H + d_ff):
        return False
    for i in range(L):
        head_count = 0
        for j in range(H):
            if arr[i, j]:
                head_count += 1
        if head_count != target_heads:
            return False
        ff_count = 0
        for j in range(d_ff):
            if arr[i, H + j]:
                ff_count += 1
        if ff_count != target_ff:
            return False
    return True
