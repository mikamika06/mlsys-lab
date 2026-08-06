def naive_matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    m, k1 = len(A), len(A[0])
    k2, n = len(B), len(B[0])
    assert k1 == k2, "Inner dimensions must agree"
    C = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0.0
            for p in range(k1):
                s += A[i][p] * B[p][j]
            C[i][j] = s
    return C
