import math

def fwht(x: list[float]) -> list[float]:
    """
    Compute the normalized fast Walsh-Hadamard transform of x (length a
    power of two), equal to (H @ x) / sqrt(n) for the recursively-built
    Hadamard matrix H, using an O(n log n) in-place butterfly — not a
    dense H @ x matrix multiply. See task.md for the exact recursion.
    """
    raise NotImplementedError('your code here')
