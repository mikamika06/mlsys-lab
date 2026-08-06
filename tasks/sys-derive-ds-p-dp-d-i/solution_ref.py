import numpy as np


def derive_ds(P: np.ndarray, dP: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    dP = np.asarray(dP, dtype=np.float64)
    
    n_rows = P.shape[0]
    n_cols = P.shape[1]
    
    result = np.empty((n_rows, n_cols), dtype=np.float64)
    
    for i in range(n_rows):
        D = 0.0
        for j in range(n_cols):
            D += P[i, j] * dP[i, j]
            
        for j in range(n_cols):
            result[i, j] = P[i, j] * (dP[i, j] - D)
            
    return result
