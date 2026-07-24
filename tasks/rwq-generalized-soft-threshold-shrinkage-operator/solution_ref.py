import numpy as np

def generalized_soft_threshold(x: np.ndarray, beta: float, p: float) -> np.ndarray:
    """Generalized soft‑thresholding for the $L_p$ quasi‑norm ($0<p\\le1$)."""
    x = np.asarray(x, dtype=np.float64)
    abs_x = np.abs(x)
    thresh = beta * np.power(abs_x, p-1)
    shrunk = np.sign(x) * np.maximum(abs_x - thresh, 0.0)
    return shrunk
