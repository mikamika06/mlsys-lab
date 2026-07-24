import numpy as np

def smoothing_transform(X: np.ndarray, W: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Incorrect implementation that swaps the order of multiplication for W'.
This will cause the product X' @ W' to differ from X @ W."""
    raise NotImplementedError('your code here')
