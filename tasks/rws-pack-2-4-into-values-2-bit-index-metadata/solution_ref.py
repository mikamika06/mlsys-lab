import numpy as np
import math

def pack_2_of_4(a: np.ndarray):
    n = a.shape[0]
    if n % 4 != 0:
        raise ValueError("Length must be divisible by 4")
    
    num_blocks = n // 4
    values_list = []
    indices_list = []
    
    for i in range(num_blocks):
        b = [a[i * 4], a[i * 4 + 1], a[i * 4 + 2], a[i * 4 + 3]]
        
        abs_b = []
        for val in b:
            if val < 0.0:
                abs_b.append(-val)
            else:
                abs_b.append(val)
        
        best1_idx = 0
        best1_val = abs_b[0]
        for j in range(1, 4):
            if abs_b[j] > best1_val:
                best1_val = abs_b[j]
                best1_idx = j
                
        best2_idx = -1
        best2_val = -1.0
        for j in range(4):
            if j == best1_idx:
                continue
            if best2_idx == -1 or abs_b[j] > best2_val:
                best2_val = abs_b[j]
                best2_idx = j
                
        if best1_idx < best2_idx:
            idx1 = best1_idx
            idx2 = best2_idx
        else:
            idx1 = best2_idx
            idx2 = best1_idx
            
        values_list.append(b[idx1])
        values_list.append(b[idx2])
        indices_list.append(idx1)
        indices_list.append(idx2)
        
    values = np.array(values_list, dtype=np.float64)
    indices = np.array(indices_list, dtype=np.uint8)
    return values, indices
