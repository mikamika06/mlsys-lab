import numpy as np


def reconstruct_from_eigh(A: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(A)
    return V @ np.diag(w) @ V.T
