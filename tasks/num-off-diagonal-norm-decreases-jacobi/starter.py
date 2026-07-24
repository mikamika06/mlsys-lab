import numpy as np


def jacobi_offdiag_norms(A: np.ndarray, n_sweeps: int) -> np.ndarray:
    """
    Run ``n_sweeps`` classical cyclic Jacobi sweeps on the symmetric matrix
    ``A`` and return the off-diagonal Frobenius norm before any sweeps and
    after each sweep, as a 1-D array of length ``n_sweeps + 1``.

    Each sweep visits pivot pairs ``(p, q)`` with ``0 <= p < q < n``, in
    row-major nested order (``p`` outer, ``q`` inner), applying exactly one
    Jacobi rotation per pair using the numerically stable formula:

        theta = (A[q,q] - A[p,p]) / (2*A[p,q])
        t     = 1                                          if theta == 0
              = sign(theta) / (|theta| + sqrt(theta**2+1))  otherwise
        c     = 1 / sqrt(t**2 + 1)
        s     = t * c

    ``A`` itself must not be mutated (work on a copy). Entries that are
    already negligible relative to the diagonal scale (below machine
    epsilon times the sum of the two diagonal magnitudes) should be treated
    as already zero and skipped, to avoid dividing by a near-zero residual.
    """
    raise NotImplementedError('your code here')
