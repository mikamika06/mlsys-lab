import numpy as np

def magnitude_optimal_2to4_mask(weights: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask that keeps the two largest‑magnitude weights in each
    consecutive block of four columns.
    """
    if weights.shape[-1] % 4 != 0:
        raise ValueError("last dimension must be a multiple of 4")
    
    flat_weights = weights.reshape(-1, 4)
    num_rows = flat_weights.shape[0]
    
    mask_flat = np.zeros((num_rows, 4), dtype=bool)
    
    for i in range(num_rows):
        row = flat_weights[i]
        abs_row = [abs(val) for val in row]
        
        max1_idx = 0
        for j in range(1, 4):
            if abs_row[j] > abs_row[max1_idx]:
                max1_idx = j
                
        max2_idx = -1
        for j in range(4):
            if j == max1_idx:
                continue
            if max2_idx == -1 or abs_row[j] > abs_row[max2_idx]:
                max2_idx = j
                
        mask_flat[i, max1_idx] = True
        mask_flat[i, max2_idx] = True
        
    return mask_flat.reshape(weights.shape)
