import numpy as np


def fold_diag_scales(W, X, s):
    M, K = W.shape
    _, N = X.shape

    W_fold = np.empty((M, K), dtype=W.dtype)
    for i in range(M):
        for j in range(K):
            W_fold[i, j] = W[i, j] * s[j]

    X_fold = np.empty((K, N), dtype=X.dtype)
    for i in range(K):
        for j in range(N):
            X_fold[i, j] = X[i, j] / s[i]

    Y_fold = np.empty((M, N), dtype=W.dtype)
    for i in range(M):
        for j in range(N):
            acc = 0.0
            for k in range(K):
                acc += W_fold[i, k] * X_fold[k, j]
            Y_fold[i, j] = acc

    max_X = 0.0
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            val = abs(X[i, j])
            if val > max_X:
                max_X = val

    max_X_fold = 0.0
    for i in range(X_fold.shape[0]):
        for j in range(X_fold.shape[1]):
            val = abs(X_fold[i, j])
            if val > max_X_fold:
                max_X_fold = val

    range_reduction_ratio = max_X / (max_X_fold + 1e-12)
    return W_fold, X_fold, Y_fold, float(range_reduction_ratio)
