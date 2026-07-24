def peak_attention_memory(N: int, heads: int, d: int) -> tuple[int, int]:
    """
    Return the peak memory usage (in bytes) of naïve and flash attention.

    Parameters
    ----------
    N : int
        Sequence length.
    heads : int
        Number of attention heads.
    d : int
        Per‑head dimensionality.

    Returns
    -------
    tuple[int, int]
        (naïve_bytes, flash_bytes)
    """
    s = 8  # bytes per float64 element
    naive = heads * N * N * s
    flash = 3 * heads * N * d * s
    return naive, flash
