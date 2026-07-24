def packed_vs_aligned_ratio(n: int) -> float:
    """
    Return the ratio of memory footprint of a naturally aligned NumPy structured
    array to that of a packed (no padding) version.
    The ratio is independent of n; it is computed from dtype.itemsize.
    """
    raise NotImplementedError('your code here')
