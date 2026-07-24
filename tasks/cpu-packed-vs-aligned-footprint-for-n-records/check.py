def grade(sol, fx) -> dict:
    """
    Compute absolute difference between learner's ratio and reference ratio.
    """
    import numpy as np

    # Reference dtypes: default align (True) vs packed (False)
    aligned_dtype = np.dtype([
        ('a', np.uint32),
        ('b', np.uint8),
        ('c', np.float64),
        ('d', np.bool_)
    ])  # align=True by default

    packed_dtype = np.dtype([
        ('a', np.uint32),
        ('b', np.uint8),
        ('c', np.float64),
        ('d', np.bool_)
    ], align=False)

    ref_ratio = aligned_dtype.itemsize / packed_dtype.itemsize

    try:
        user_ratio = sol.packed_vs_aligned_ratio(123456)  # any positive n
    except Exception:
        return {"ratio_diff": float("inf")}

    diff = abs(user_ratio - ref_ratio)
    return {"ratio_diff": diff}
