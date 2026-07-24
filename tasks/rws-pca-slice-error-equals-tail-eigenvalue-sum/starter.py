import numpy as np


def pca_slice_error(X: np.ndarray, k: int) -> float:
    """Squared Frobenius error of projecting X onto the top-k eigenvectors
    of the Gram matrix G = X^T X (no centering): ||X - X Q_k Q_k^T||_F^2.
    """
    raise NotImplementedError('your code here')
