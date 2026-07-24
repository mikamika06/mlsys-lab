import numpy as np

def cosine_similarity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # TODO: missing normalization – this returns raw dot products.
    # This will produce incorrect rankings and may raise errors for zero vectors.
    return A @ B.T
