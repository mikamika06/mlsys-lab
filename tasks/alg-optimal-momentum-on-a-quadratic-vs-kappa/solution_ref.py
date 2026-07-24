import numpy as np

def optimal_momentum_beta(A: np.ndarray) -> float:
    """
    Compute the optimal momentum coefficient for gradient descent on a quadratic
    with Hessian A, given by ((sqrt(kappa)-1)/(sqrt(kappa)+1))^2,
    where kappa is the condition number of A.
    """
    if not isinstance(A, np.ndarray):
        raise ValueError("Input must be a NumPy array.")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")
    # Ensure symmetry
    if not np.allclose(A, A.T, atol=1e-8):
        raise ValueError("A must be symmetric.")
    eigs = np.linalg.eigvalsh(A)
    if np.any(eigs <= 0):
        raise ValueError("A must be positive-definite.")
    kappa = eigs[-1] / eigs[0]
    beta = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** 2
    return float(beta)
