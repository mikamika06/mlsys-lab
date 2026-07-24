import numpy as np


def fwht(x: np.ndarray) -> np.ndarray:
    """
    Normalized fast Walsh-Hadamard transform, O(n log n) butterfly.

    Equivalent to (H @ x) / sqrt(n) where H is the n x n unnormalized
    Hadamard matrix built by the recursive block rule
    H_1 = [[1]], H_{2m} = [[H_m, H_m], [H_m, -H_m]].
    """
    x = np.asarray(x, dtype=np.float64).copy()
    n = x.shape[0]
    if n & (n - 1) != 0:
        raise ValueError("length must be a power of two")

    h = 1
    while h < n:
        x = x.reshape(-1, 2, h)
        a = x[:, 0, :].copy()
        b = x[:, 1, :].copy()
        x[:, 0, :] = a + b
        x[:, 1, :] = a - b
        x = x.reshape(n)
        h *= 2

    return x / np.sqrt(n)
