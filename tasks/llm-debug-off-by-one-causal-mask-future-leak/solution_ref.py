import numpy as np

def create_causal_mask(n: int) -> np.ndarray:
    """
    Return an (n, n) causal mask with ones on and below the main diagonal.
    The result is a float64 NumPy array.
    """
    return np.tril(np.ones((n, n), dtype=np.float64))
