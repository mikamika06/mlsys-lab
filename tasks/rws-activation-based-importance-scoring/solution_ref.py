import numpy as np

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
    # Ensure we work with a NumPy array and use float64 precision
    activations = np.asarray(activations, dtype=np.float64)
    # Compute mean of absolute values along the batch axis (axis=0)
    importance = np.mean(np.abs(activations), axis=0)
    return importance.astype(np.float64)
