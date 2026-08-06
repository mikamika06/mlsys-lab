import numpy as np

def migration_channels(X, threshold):
    """Return sorted list of channel indices whose absmax > threshold * median_absmax."""
    X = np.asarray(X, dtype=np.float64)
    n_rows = X.shape[0]
    n_cols = X.shape[1]
    
    channel_absmax = []
    for j in range(n_cols):
        current_max = 0.0
        for i in range(n_rows):
            val = X[i, j]
            if val < 0.0:
                val = -val
            if val > current_max:
                current_max = val
        channel_absmax.append(current_max)
        
    sorted_absmax = sorted(channel_absmax)
    n = len(sorted_absmax)
    if n % 2 == 1:
        rho = sorted_absmax[n // 2]
    else:
        rho = (sorted_absmax[n // 2 - 1] + sorted_absmax[n // 2]) / 2.0
        
    cutoff = threshold * rho
    flagged = []
    for j in range(n_cols):
        if channel_absmax[j] > cutoff:
            flagged.append(j)
            
    return sorted(int(i) for i in flagged)
