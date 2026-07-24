from __future__ import annotations

import numpy as np


def inverse_iteration(A: np.ndarray, num_iters: int = 100, x0: np.ndarray | None = None):
    """Inverse iteration for the eigenpair of smallest |eigenvalue|.

    Parameters
    ----------
    A : (n, n) ndarray
        A square, invertible (in the tests: symmetric positive-definite)
        matrix.
    num_iters : int
        Number of inverse-iteration steps to run.
    x0 : (n,) ndarray or None
        Optional starting vector; if None, use a uniform vector.

    Returns
    -------
    (eigval, eigvec) : tuple[float, np.ndarray]
        The approximate smallest-|eigenvalue| eigenpair of ``A``.
        ``eigvec`` should be unit-norm.

    Each step should solve ``A y = x`` for ``y`` (NOT multiply by ``A``),
    then renormalize ``x = y / ||y||``. After ``num_iters`` steps, return
    the Rayleigh quotient ``x @ A @ x`` as the eigenvalue estimate.
    """
    raise NotImplementedError('your code here')
