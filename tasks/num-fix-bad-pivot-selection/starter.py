def lu_partial_pivot(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    """
    Factor a square matrix ``A`` as ``P @ A = L @ U``.

    * ``P`` — n x n permutation matrix (float64, entries 0.0/1.0).
    * ``L`` — n x n unit lower-triangular matrix (ones on the diagonal).
    * ``U`` — n x n upper-triangular matrix.

    BUG: this picks the "first nonzero pivot" — it only swaps rows when the
    current diagonal candidate is EXACTLY zero, instead of searching for the
    largest-magnitude candidate. On an ill-scaled matrix (a tiny-but-nonzero
    pivot with much larger entries below it) this produces huge multipliers
    and the L/U factors lose almost all precision, even though A[k, k] != 0
    technically holds.
    """
    raise NotImplementedError('your code here')
