def pairwise_l2_matrix(
    X: list[list[float]], Y: list[list[float]]
) -> list[list[float]]:
    n = len(X)
    d = len(X[0]) if n > 0 else 0
    m = len(Y)

    X_norm = [0.0] * n
    for i in range(n):
        s = 0.0
        for k in range(d):
            s += float(X[i][k]) ** 2
        X_norm[i] = s

    Y_norm = [0.0] * m
    for j in range(m):
        s = 0.0
        for k in range(d):
            s += float(Y[j][k]) ** 2
        Y_norm[j] = s

    result = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            dot = 0.0
            for k in range(d):
                dot += float(X[i][k]) * float(Y[j][k])
            result[i][j] = X_norm[i] + Y_norm[j] - 2.0 * dot

    return result
