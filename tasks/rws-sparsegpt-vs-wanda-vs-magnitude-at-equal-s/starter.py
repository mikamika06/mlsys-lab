import numpy as np


def compare_prune_methods_mse(W: np.ndarray, X: np.ndarray, sparsity: float, lam: float) -> dict:
    """W:(m,d), X:(d,n). remove=int(m*d*sparsity) weights, globally, for
    each method: magnitude (smallest |W|), wanda (smallest |W_ij|*z_j,
    z_j=norm(X[j,:])), sparsegpt (smallest w^2/Hinv[j,j] with
    H=2*X@X.T+lam*I, OBS row-compensation after each removal). Return
    {"mse_magnitude","mse_wanda","mse_sparsegpt"}, each
    mean((W@X - Wq@X)**2)."""
    raise NotImplementedError('your code here')
