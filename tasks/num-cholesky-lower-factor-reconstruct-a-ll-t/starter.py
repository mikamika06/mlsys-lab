import numpy as np


def cholesky_lower(A: np.ndarray) -> np.ndarray:
    """Return the lower-triangular Cholesky factor L of a symmetric positive definite A.

    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        Symmetric positive definite matrix.

    Returns
    -------
    L : np.ndarray, shape (n, n), float64
        Lower triangular (strict upper part exactly zero) with L[i, i] > 0,
        satisfying ``L @ L.T == A``.

    Note: ``np.linalg.cholesky`` (and the SciPy equivalents) are detected by the
    grader and will fail the run. Implement the factorisation yourself.
    """
    raise NotImplementedError('your code here')
