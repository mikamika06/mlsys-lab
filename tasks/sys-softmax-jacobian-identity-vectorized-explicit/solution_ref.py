import numpy as np


def softmax_jacobian_vjp(p: np.ndarray, dY: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    dY = np.asarray(dY, dtype=np.float64)
    
    rows, cols = p.shape
    out = np.empty((rows, cols), dtype=np.float64)
    
    for i in range(rows):
        dot = 0.0
        for j in range(cols):
            dot += p[i, j] * dY[i, j]
        for j in range(cols):
            out[i, j] = p[i, j] * (dY[i, j] - dot)
            
    return out
