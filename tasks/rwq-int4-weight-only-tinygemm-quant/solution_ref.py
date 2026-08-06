import math
import numpy as np


def tinygemm_int4_quantize(W: np.ndarray, group_size: int = 128):
    W_float = np.asarray(W, dtype=np.float64)
    rows, cols = W_float.shape
    ng = cols // group_size

    q = np.empty((rows, cols), dtype=np.uint8)
    scale = np.empty((rows, ng), dtype=np.float64)
    zero_point = np.empty((rows, ng), dtype=np.float64)
    deq = np.empty((rows, cols), dtype=np.float64)

    for i in range(rows):
        for g in range(ng):
            gmin = W_float[i, g * group_size]
            gmax = W_float[i, g * group_size]
            
            for k in range(1, group_size):
                val = W_float[i, g * group_size + k]
                if val < gmin:
                    gmin = val
                if val > gmax:
                    gmax = val
                    
            s = (gmax - gmin) / 15.0
            if s == 0.0:
                s_safe = 1.0
            else:
                s_safe = s
                
            scale[i, g] = s
            zero_point[i, g] = gmin
            
            for k in range(group_size):
                c = g * group_size + k
                val = W_float[i, c]
                q_val = (val - gmin) / s_safe
                
                floor_val = math.floor(q_val)
                diff = q_val - floor_val
                
                if diff < 0.5:
                    q_int = floor_val
                elif diff > 0.5:
                    q_int = floor_val + 1
                else:
                    if floor_val % 2 == 0:
                        q_int = floor_val
                    else:
                        q_int = floor_val + 1
                        
                if q_int < 0:
                    q_int = 0
                elif q_int > 15:
                    q_int = 15
                    
                q[i, c] = q_int
                deq[i, c] = float(q_int) * s_safe + gmin

    return q, scale, zero_point, deq
