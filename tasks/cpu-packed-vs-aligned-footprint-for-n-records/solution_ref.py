def packed_vs_aligned_ratio(n: int) -> float:
    """
    Return the ratio of memory footprint of a naturally aligned NumPy structured
    array to that of a packed (no padding) version.
    The ratio is independent of n; it is computed from dtype.itemsize.
    """
    import numpy as np

    # Define the two dtypes
    aligned_dtype = np.dtype([
        ('a', np.uint32),
        ('b', np.uint8),
        ('c', np.float64),
        ('d', np.bool_)
    ])  # align=True by default on most platforms

    packed_dtype = np.dtype([
        ('a', np.uint32),
        ('b', np.uint8),
        ('c', np.float64),
        ('d', np.bool_)
    ], align=False)

    # Ratio of total sizes
    ratio = aligned_dtype.itemsize / packed_dtype.itemsize
    return float(ratio)
