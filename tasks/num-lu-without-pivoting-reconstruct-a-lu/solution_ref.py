def lu_no_pivot(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    n = len(A)

    L = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    U = [[0.0 for j in range(n)] for i in range(n)]

    for k in range(n):
        for j in range(k, n):
            dot_product = 0.0
            for m in range(k):
                dot_product += L[k][m] * U[m][j]
            U[k][j] = A[k][j] - dot_product

        for i in range(k + 1, n):
            dot_product = 0.0
            for m in range(k):
                dot_product += L[i][m] * U[m][k]
            L[i][k] = (A[i][k] - dot_product) / U[k][k]

    return L, U
