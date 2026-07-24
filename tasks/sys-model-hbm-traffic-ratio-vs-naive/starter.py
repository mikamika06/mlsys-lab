def hbm_traffic(N: int, d: int, M: int, elem_bytes: int = 4) -> dict:
    """Model HBM (off-chip memory) traffic, in elements and bytes, for naive
    vs tiled (FlashAttention-style) single-head attention.

    See task.md for the exact naive and tiled traffic formulas.

    Returns
    -------
    dict with keys:
      naive_bytes, tiled_bytes : int -- total HBM bytes moved by each scheme.
      size_ratio : float -- tiled_bytes / naive_bytes.
    """
    raise NotImplementedError('your code here')
