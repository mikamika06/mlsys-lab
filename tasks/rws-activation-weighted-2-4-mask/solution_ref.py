import math
import numpy as np


def activation_weighted_2_4_mask(W: np.ndarray, X: np.ndarray):
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    rows_W = W.shape[0]
    cols_W = W.shape[1]
    rows_X = X.shape[0]
    cols_X = X.shape[1]

    scales = [
        math.sqrt(sum(float(X[i, j]) * float(X[i, j]) for j in range(cols_X)))
        for i in range(rows_X)
    ]

    scores = [
        [abs(float(W[i, j])) * scales[j] for j in range(cols_W)]
        for i in range(rows_W)
    ]

    mask_list = [[0 for _ in range(cols_W)] for _ in range(rows_W)]
    for i in range(rows_W):
        for start in range(0, cols_W, 4):
            chunk = [scores[i][start + k] for k in range(4)]
            indexed = sorted(enumerate(chunk), key=lambda x: x[1])
            keep = [indexed[2][0], indexed[3][0]]
            for idx in keep:
                mask_list[i][start + idx] = 1

    A = [
        [sum(float(W[i, k]) * float(X[k, j]) for k in range(cols_W)) for j in range(cols_X)]
        for i in range(rows_W)
    ]

    B = [
        [
            sum((float(W[i, k]) * float(mask_list[i][k])) * float(X[k, j]) for k in range(cols_W))
            for j in range(cols_X)
        ]
        for i in range(rows_W)
    ]

    error_acc = 0.0
    for i in range(rows_W):
        for j in range(cols_X):
            diff = A[i][j] - B[i][j]
            error_acc += diff * diff

    mask = np.array(mask_list, dtype=np.int64)
    return mask, float(error_acc)
