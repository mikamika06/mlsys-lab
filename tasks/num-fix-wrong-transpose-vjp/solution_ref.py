def matmul_vjp(A: list[list[float]], B: list[list[float]], G: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    m = len(A)
    k = len(A[0])
    n = len(B[0])

    # B_T: n x k
    B_T = [[B[r][c] for r in range(k)] for c in range(n)]

    # dA = G @ B.T -> m x n @ n x k = m x k
    dA = [[sum(G[i][p] * B_T[p][j] for p in range(n)) for j in range(k)] for i in range(m)]

    # A_T: k x m
    A_T = [[A[r][c] for r in range(m)] for c in range(k)]

    # dB = A.T @ G -> k x m @ m x n = k x n
    dB = [[sum(A_T[i][p] * G[p][j] for p in range(m)) for j in range(n)] for i in range(k)]

    return dA, dB
