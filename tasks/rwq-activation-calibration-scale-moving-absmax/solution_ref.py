import numpy as np
import math

def moving_absmax(batches, momentum):
    """
    Correct implementation of the moving absolute maximum calibration.
    """
    num_cols = batches[0].shape[1]
    scale = np.zeros(num_cols, dtype=np.float64)
    
    for batch in batches:
        num_rows = batch.shape[0]
        absmax = np.zeros(num_cols, dtype=np.float64)
        
        for j in range(num_cols):
            current_max = 0.0
            for i in range(num_rows):
                val = batch[i, j]
                if val < 0.0:
                    abs_val = -val
                else:
                    abs_val = val
                if i == 0 or abs_val > current_max:
                    current_max = abs_val
            absmax[j] = current_max
            
        for j in range(num_cols):
            scale[j] = momentum * scale[j] + (1 - momentum) * absmax[j]
            
    return scale
