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
        # dot product of the i‑th row with x
        y[i] = np.dot(data[start:end], x[indices[start:end]])
    return y
