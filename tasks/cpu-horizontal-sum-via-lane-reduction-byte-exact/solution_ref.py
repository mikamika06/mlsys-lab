def lane_reduce_sum(a: list[int]) -> int:
    """
    Compute the horizontal sum of a list of integers via SIMD‑style lane reduction.
    The result is returned as an integer.
    """
    if not isinstance(a, list):
        raise ValueError("Input must be a list.")
    total = 0
    for i in range(len(a)):
        total += a[i]
    return total
