import numpy as np

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, *, causal: bool=False) -> np.ndarray:
    """Incorrect implementation: missing scaling and uses a wrong mask."""
    raise NotImplementedError('your code here')
