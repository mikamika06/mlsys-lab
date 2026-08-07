def fold_smoothing(X: list[list[float]], W: list[list[float]], s: list[float]) -> tuple[list[list[float]], list[list[float]]]:
    m = len(X)
    d = len(s)
    n = len(W[0]) if W else 0

    X_new = []
    for i in range(m):
        row = []
        for j in range(d):
            row.append(X[i][j] * s[j])
        X_new.append(row)

    W_new = []
    for i in range(d):
        row = []
        for j in range(n):
            row.append(W[i][j] / s[i])
        W_new.append(row)

    return X_new, W_new
