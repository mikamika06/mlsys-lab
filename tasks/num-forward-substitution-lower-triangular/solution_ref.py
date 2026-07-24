import numpy as np


def forward_sub(L, b):
    """Solve Lx = b where L is lower triangular (n×n) with a nonzero
    diagonal, by rows. Returns a float64 NumPy array of length n."""
    L = np.asarray(L, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = len(b)
    x = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = b[i]
        for j in range(i):
            s -= L[i, j] * x[j]
        x[i] = s / L[i, i]
    return x
