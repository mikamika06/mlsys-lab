import numpy as np
import math

def score_importance(activations: np.ndarray) -> np.ndarray:
    """
    Compute mean absolute activation per unit.

    Parameters
    ----------
    activations : np.ndarray
        2‑D array of shape (batch, units). The values may be positive or negative.
    
    Returns
    -------
    importance : np.ndarray
        1‑D float64 array of length ``units`` containing the mean absolute activation for each unit.
    """
    activations = np.asarray(activations, dtype=np.float64)
    batch_size = activations.shape[0]
    units = activations.shape[1]
    
    importance = np.zeros(units, dtype=np.float64)
    for j in range(units):
        total = 0.0
        for i in range(batch_size):
            val = activations[i, j]
            if val < 0.0:
                val = -val
            total += val
        if batch_size > 0:
            importance[j] = total / batch_size
        else:
            importance[j] = 0.0
            
    return importance.astype(np.float64)
