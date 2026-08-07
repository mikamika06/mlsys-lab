def fix_awq_scale(W: list[list[float]], X: list[list[float]], s: list[float]) -> list[list[float]]:
    m = len(W)
    d = len(s)
    n = len(X)

    W_scaled = [[W[i][j] * s[j] for j in range(d)] for i in range(m)]
    X_fixed = [[X[i][j] / s[j] for j in range(d)] for i in range(n)]

    Y = []
    for i in range(n):
        row = []
        for k in range(m):
            val = sum(X_fixed[i][j] * W_scaled[k][j] for j in range(d))
            row.append(val)
        Y.append(row)
    return Y
