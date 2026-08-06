import numpy as np

def csr_matvec(data, indices, indptr, x):
    """
    Multiply a matrix in CSR format by a dense vector.
    Returns a 1‑D NumPy array of float64 containing the result.
    """
    n = len(indptr) - 1
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = indptr[i]
        end = indptr[i + 1]
        total = 0.0
        for j in range(start, end):
            total += data[j] * x[indices[j]]
        y[i] = total
    return y
