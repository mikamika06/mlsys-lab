import numpy as np
import math

def magnitude_prune_mask(weights: np.ndarray, keep_fraction: float) -> np.ndarray:
    """
    Return a boolean mask selecting the top‑`keep_fraction` fraction of weights by absolute value.
    Ties are broken stably according to the original index order.
    """
    n = len(weights)
    k = int(math.ceil(keep_fraction * n))
    if k <= 0:
        return np.zeros(n, dtype=bool)
    
    abs_weights = []
    for i in range(n):
        w = weights[i]
        val = -w if w < 0 else w
        abs_weights.append((val, i))
    
    def merge_sort(arr):
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = merge_sort(arr[:mid])
        right = merge_sort(arr[mid:])
        
        merged = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i][0] > right[j][0] or (left[i][0] == right[j][0] and left[i][1] <= right[j][1]):
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
        while i < len(left):
            merged.append(left[i])
            i += 1
        while j < len(right):
            merged.append(right[j])
            j += 1
        return merged

    sorted_abs = merge_sort(abs_weights)
    
    mask = np.zeros(n, dtype=bool)
    for i in range(k):
        orig_idx = sorted_abs[i][1]
        mask[orig_idx] = True
        
    return mask
