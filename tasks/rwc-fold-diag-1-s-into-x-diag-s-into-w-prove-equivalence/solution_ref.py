def fold_diag_scales(W, X, s):
    M = len(W)
    K = len(W[0])
    _, N = len(X), len(X[0])

    W_fold = [[0.0] * K for _ in range(M)]
    for i in range(M):
        for j in range(K):
            W_fold[i][j] = W[i][j] * s[j]

    X_fold = [[0.0] * N for _ in range(K)]
    for i in range(K):
        for j in range(N):
            X_fold[i][j] = X[i][j] / s[i]

    Y_fold = [[0.0] * N for _ in range(M)]
    for i in range(M):
        for j in range(N):
            acc = 0.0
            for k in range(K):
                acc += W_fold[i][k] * X_fold[k][j]
            Y_fold[i][j] = acc

    max_X = 0.0
    for i in range(len(X)):
        for j in range(len(X[0])):
            val = abs(X[i][j])
            if val > max_X:
                max_X = val

    max_X_fold = 0.0
    for i in range(len(X_fold)):
        for j in range(len(X_fold[0])):
            val = abs(X_fold[i][j])
            if val > max_X_fold:
                max_X_fold = val

    range_reduction_ratio = max_X / (max_X_fold + 1e-12)
    return W_fold, X_fold, Y_fold, float(range_reduction_ratio)
