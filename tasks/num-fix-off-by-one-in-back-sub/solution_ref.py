import numpy as np

def back_sub(U, b):
    """Solve Ux = b where U is upper triangular (n×n). Returns array x of length n."""
    n = len(b)
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= U[i, j] * x[j]
        x[i] /= U[i, i]
    return x
