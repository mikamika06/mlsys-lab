import numpy as np

def add_residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Add two tensors and return the result."""
    return a + b

def transformer_block(x: np.ndarray,
                      w1: np.ndarray, b1: np.ndarray,
                      w2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Compute one transformer block with two residual connections.
    All operations are vectorised; the helper `add_residual` is used exactly twice.
    """
    a = x @ w1 + b1
    y = add_residual(x, a)
    b = y @ w2 + b2
    z = add_residual(y, b)
    return z.astype("float64")
