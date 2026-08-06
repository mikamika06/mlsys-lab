import numpy as np
import math


def gumbel_softmax_relaxed(logits: np.ndarray, g: np.ndarray, tau: float) -> np.ndarray:
    """
    Gumbel-softmax relaxation with externally supplied (fixed) Gumbel
    noise g: softmax((logits + g) / tau, axis=-1), computed in a
    numerically stable way (subtract the row max before exponentiating).
    """
    logits = np.asarray(logits, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    
    shape = logits.shape
    last_dim = shape[-1]
    
    flat_logits = logits.reshape(-1, last_dim)
    flat_g = g.reshape(-1, last_dim)
    n_rows, n_cols = flat_logits.shape
    
    out_flat = np.empty((n_rows, n_cols), dtype=np.float64)
    
    for i in range(n_rows):
        z_row = [
            (flat_logits[i, j] + flat_g[i, j]) / tau
            for j in range(n_cols)
        ]
        
        max_z = z_row[0]
        for j in range(1, n_cols):
            if z_row[j] > max_z:
                max_z = z_row[j]
                
        e_row = []
        for j in range(n_cols):
            e_row.append(math.exp(z_row[j] - max_z))
            
        sum_e = 0.0
        for val in e_row:
            sum_e += val
            
        for j in range(n_cols):
            out_flat[i, j] = e_row[j] / sum_e
            
    return out_flat.reshape(shape)
