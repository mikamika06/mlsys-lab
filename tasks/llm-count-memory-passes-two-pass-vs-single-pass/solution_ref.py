def count_memory_passes(N: int, D: int) -> tuple[int, int]:
    """
    Return the number of element reads for a naïve two‑pass LayerNorm
    and an optimised single‑pass implementation.

    Parameters
    ----------
    N : int
        Number of samples (rows).
    D : int
        Feature dimension (columns).

    Returns
    -------
    tuple[int, int]
        (two_pass_reads, single_pass_reads)
    """
    two_pass = 2 * N * D
    single_pass = N * D
    return two_pass, single_pass
