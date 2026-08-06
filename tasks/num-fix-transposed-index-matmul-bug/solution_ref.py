def matmul_naive(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Compute A @ B for A of shape (m,k) and B of shape (k,n) using explicit
    Python loops. Returns a list of lists of floats of shape (m,n)."""
    m = len(A)
    k = len(A[0])
    k2 = len(B)
    n = len(B[0])
    C = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0.0
            for p in range(k):
                s += A[i][p] * B[p][j]
            C[i][j] = s
    return C
