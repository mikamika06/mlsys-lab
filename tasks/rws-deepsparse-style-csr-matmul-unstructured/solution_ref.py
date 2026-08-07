def csr_matmul(data, indices, indptr, X):
    m = len(indptr) - 1
    n = len(X[0])
    Y = [[0.0 for _ in range(n)] for _ in range(m)]

    for row in range(m):
        start = int(indptr[row])
        end = int(indptr[row + 1])
        for p in range(start, end):
            col = int(indices[p])
            val = float(data[p])
            for j in range(n):
                Y[row][j] += val * X[col][j]
    return Y
