import numpy as np

def recover_mask(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return P > eps
