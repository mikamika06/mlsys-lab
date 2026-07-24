def mixed_precision_memory_accounting(H: int, k: int) -> tuple[float, float]:
    """
    Compute effective bytes per row-group and the ratio to a fully fp16 implementation.

    Parameters
    ----------
    H : int
        Total number of columns (hidden dimension).
    k : int
        Number of outlier columns kept at 2‑byte precision.
        The remaining H-k columns are compressed to 1 byte each.

    Returns
    -------
    bytes_per_row_group : float
        Bytes required per row-group: (H - k) * 1 + k * 2.
    size_ratio : float
        Ratio of the computed bytes to the baseline fp16 memory usage:
        ((H - k) * 1 + k * 2) / (2 * H).
    """
    if not isinstance(H, int) or not isinstance(k, int):
        raise TypeError("H and k must be integers")
    if H < 0 or k < 0 or k > H:
        raise ValueError("Invalid values: require 0 <= k <= H")

    bytes_per_row_group = (H - k) * 1 + k * 2
    size_ratio = bytes_per_row_group / (2.0 * H)
    return float(bytes_per_row_group), float(size_ratio)
