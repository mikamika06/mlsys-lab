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
    B, T, C = X.shape
    out = np.zeros(C, dtype=X.dtype)
    total_elements = B * T
    for c in range(C):
        acc = 0.0
        for b in range(B):
            for t in range(T):
                val = X[b, t, c]
                if val < 0.0:
                    val = -val
                acc += val
        out[c] = acc / total_elements
    return out
