def restore_awq_equivalence(
    X: list[list[float]], W: list[list[float]], s: list[float]
) -> list[list[float]]:
    n = len(X)
    d = len(s)
    m = len(W)

    X_comp = [[X[i][j] / s[j] for j in range(d)] for i in range(n)]
    W_scaled = [[W[i][j] * s[j] for j in range(d)] for i in range(m)]

    Y = []
    for i in range(n):
        row = []
        for j in range(m):
            val = 0.0
            for k in range(d):
                val += X_comp[i][k] * W_scaled[j][k]
            row.append(val)
        Y.append(row)

    return Y
