import numpy as np

def masked_sum(data, mask):
    """Return the sum of data values where mask is True.

    Uses np.where to avoid the 0*inf NaN trap that data*mask creates.
    """
    data = np.asarray(data, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    
    total = 0.0
    for i in range(len(data)):
        if mask[i]:
            total += data[i]
            
    return float(total)
