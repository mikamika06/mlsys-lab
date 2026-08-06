import math

def pairwise_mahalanobis(X: list[list[float]], cov_inv: list[list[float]]) -> list[list[float]]:
    n = len(X)
    d = len(X[0])

    Y = [[0.0] * d for _ in range(n)]
    for i in range(n):
        for k in range(d):
            s = 0.0
            for j in range(d):
                s += X[i][j] * cov_inv[j][k]
            Y[i][k] = s

    diag = [0.0] * n
    for i in range(n):
        s = 0.0
        for k in range(d):
            s += X[i][k] * Y[i][k]
        diag[i] = s

    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            xy = 0.0
            for k in range(d):
                xy += X[i][k] * Y[j][k]
            d2 = diag[i] + diag[j] - 2.0 * xy
            if d2 < 0.0:
                d2 = 0.0
            out[i][j] = math.sqrt(d2)

    return out
