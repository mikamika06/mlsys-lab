import numpy as np


def csr_matmul(data, indices, indptr, X):
    m = len(indptr) - 1
    n = X.shape[1]
    Y = np.zeros((m, n), dtype=np.float64)

    for row in range(m):
        start = int(indptr[row])
        end = int(indptr[row + 1])
        for p in range(start, end):
            Y[row] += float(data[p]) * X[int(indices[p])]
    return Y
