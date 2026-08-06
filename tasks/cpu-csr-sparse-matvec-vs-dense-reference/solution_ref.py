def csr_matvec(data: list[float], indices: list[int], indptr: list[int], x: list[float]) -> list[float]:
    """
    Multiply a matrix in CSR format by a dense vector.
    Returns a list of floats containing the result.
    """
    n = len(indptr) - 1
    y = [0.0] * n
    for i in range(n):
        start = indptr[i]
        end = indptr[i + 1]
        total = 0.0
        for j in range(start, end):
            total += data[j] * x[indices[j]]
        y[i] = total
    return y
