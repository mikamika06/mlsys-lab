import numpy as np

def per_input_channel_importance(X: np.ndarray) -> np.ndarray:
    """
    Compute the mean absolute activation for each input channel.

    Parameters
    ----------
    X : np.ndarray
        Input tensor of shape (B, T, C).

    Returns
    -------
    np.ndarray
        1‑D array of length C containing the per‑channel importance.
    """
    return np.mean(np.abs(X), axis=(0, 1))
