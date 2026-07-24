import numpy as np


def block_power_topk(A: np.ndarray, Q0: np.ndarray, n_iter: int):
    """Subspace (block power) iteration for the top-k eigenpairs of a symmetric matrix.

    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        Symmetric matrix.
    Q0 : np.ndarray, shape (n, k)
        Starting block of k column vectors (not necessarily orthonormal).
    n_iter : int
        Number of block iterations to perform.

    Returns
    -------
    (eigvals, Q) : tuple[np.ndarray, np.ndarray]
        ``eigvals`` has shape (k,), sorted in descending order.
        ``Q`` has shape (n, k); column j is the unit eigenvector for ``eigvals[j]``.
    """
    raise NotImplementedError('your code here')
