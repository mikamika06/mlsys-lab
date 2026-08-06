import numpy as np
import math

def compute_activation_scale(X: np.ndarray) -> np.ndarray:
    """
    Compute per‑channel mean absolute activation.

    Parameters
    ----------
    X : np.ndarray
        3‑D array of shape (batch, seq_len, channels).

    Returns
    -------
    np.ndarray
        1‑D float64 array of length `channels` containing the statistic.
    """
    batch_size, seq_len, channels = X.shape
    total_elements = batch_size * seq_len
    
    result = np.zeros(channels, dtype=np.float64)
    
    for c in range(channels):
        acc = 0.0
        for b in range(batch_size):
            for s in range(seq_len):
                val = X[b, s, c]
                if val < 0.0:
                    acc -= val
                else:
                    acc += val
        result[c] = acc / total_elements
        
    return result
