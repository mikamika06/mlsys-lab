import math


def fwht(x: list[float]) -> list[float]:
    """
    Normalized fast Walsh-Hadamard transform, O(n log n) butterfly.

    Equivalent to (H @ x) / sqrt(n) where H is the n x n unnormalized
    Hadamard matrix built by the recursive block rule
    H_1 = [[1]], H_{2m} = [[H_m, H_m], [H_m, -H_m]].
    """
    x = list(x)
    n = len(x)
    if n & (n - 1) != 0:
        raise ValueError("length must be a power of two")

    h = 1
    while h < n:
        for i in range(0, n, 2 * h):
            for j in range(h):
                idx0 = i + j
                idx1 = i + j + h
                a = x[idx0]
                b = x[idx1]
                x[idx0] = a + b
                x[idx1] = a - b
        h *= 2

    inv_sqrt_n = 1.0 / math.sqrt(n)
    for i in range(n):
        x[i] *= inv_sqrt_n

    return x
