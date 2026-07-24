import numpy as np

def compute_int8_scales(X: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-row X scales and per-column W scales for int8 quantization.

    Args:
        X: activation matrix of shape (n, d)
        W: weight matrix of shape (d, m)

    Returns:
        scale_x: array of shape (n,) with max(|x_i|) / 127 for each row
        scale_w: array of shape (m,) with max(|w_j|) / 127 for each column
    """
    raise NotImplementedError("Implement compute_int8_scales")
