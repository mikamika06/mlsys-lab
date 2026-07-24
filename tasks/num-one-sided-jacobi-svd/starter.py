import numpy as np

def one_sided_jacobi_svd(A, tol=1e-12, max_iter=1000):
    """
    Compute singular values of A via the one-sided Jacobi SVD algorithm.

    Parameters
    ----------
    A : np.ndarray of shape (n, n), square real matrix.
    tol : float, convergence tolerance on off-diagonal Frobenius norm.
    max_iter : int, maximum number of Jacobi sweeps.

    Returns
    -------
    singular_values : np.ndarray of shape (n,), sorted in descending order.
    """
    raise NotImplementedError("your code here")
