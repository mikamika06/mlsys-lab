import numpy as np

def scale_invariant_product(W: np.ndarray, X: np.ndarray, s: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the original product and the product after scaling weights by *s*
    and activations by 1/s.
    """
    orig = W @ X
    scaled = (W * s) @ (X / s)
    return orig, scaled
