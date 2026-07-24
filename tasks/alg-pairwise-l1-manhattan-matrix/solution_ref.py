import numpy as np

def pairwise_l1_matrix(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    """
    Compute the pairwise Manhattan (L1) distance matrix between rows of X and Y.
    If Y is None, compute distances within X.

    Parameters
    ----------
    X : np.ndarray
        2-D array of shape (n, d).
    Y : np.ndarray | None, optional
        2-D array of shape (m, d). Defaults to None.

    Returns
    -------
    D : np.ndarray
        2-D array of shape (n, m) containing L1 distances.
    """
    if Y is None:
        Y = X
    # Broadcasting difference and summing absolute values over the feature axis
    return np.abs(X[:, None, :] - Y[None, :, :]).sum(axis=2)
