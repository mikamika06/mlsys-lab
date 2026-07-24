import numpy as np


def search_best_alpha(W: np.ndarray, X: np.ndarray, alphas: np.ndarray):
    """
    Grid-search the SmoothQuant migration hyper-parameter `alpha`.

    Parameters
    ----------
    W : np.ndarray
        Weight tensor of shape (C_out, *).
    X : np.ndarray
        Activation tensor of shape (N, C_out, *).
    alphas : np.ndarray
        1-D array of candidate alpha values in [0, 1].

    Returns
    -------
    best_idx : int
        Index into `alphas` of the alpha that minimises
        max(activation INT8 rel error, weight INT8 rel error) after migration.
    errors : np.ndarray
        1-D float64 array of length len(alphas); errors[k] is that
        max-of-two-errors value for alphas[k].
    """
    raise NotImplementedError('your code here')
