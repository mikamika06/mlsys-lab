import numpy as np

def prune(w: np.ndarray, p: float) -> np.ndarray:
    """Zero out the lowest magnitude fraction `p` of weights."""
    raise NotImplementedError

def quantize(w: np.ndarray, b: int) -> np.ndarray:
    """Asymmetric min-max quantization to `b` bits."""
    raise NotImplementedError
