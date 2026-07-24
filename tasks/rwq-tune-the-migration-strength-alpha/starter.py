import numpy as np


def sweep_alpha(W: np.ndarray, X: np.ndarray, alphas: np.ndarray):
    """
    Sweep the SmoothQuant migration-strength grid `alphas` and return
    (best_idx, best_mse): the index of the alpha minimizing the W8A8
    output MSE, and that MSE, as described in task.md.
    """
    raise NotImplementedError('your code here')
