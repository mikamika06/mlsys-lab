import numpy as np


def pca_projection(X: np.ndarray, k: int) -> np.ndarray:
    # TODO: incorrectly uses U as if it contained feature directions.
    # U has sample-space directions, so this returns the wrong coordinates.
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    return u[:, :k]
