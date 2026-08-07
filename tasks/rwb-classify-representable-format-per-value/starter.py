def classify_better_format(values: list[float]) -> list[int]:
    """
    For each scalar in `values`, round-trip it through both E4M3 and E5M2
    (nearest representable value, clamped at the format's finite max
    magnitude, no rescaling), and label which format gives the smaller
    absolute round-trip error: 0 -> E4M3 (or exact tie), 1 -> E5M2.
    Returns an int array of the same shape as `values`.
    """
    raise NotImplementedError('your code here')
