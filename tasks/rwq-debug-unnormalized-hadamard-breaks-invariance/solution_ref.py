import math


def _hadamard(n):
    H = [[1.0]]
    while len(H) < n:
        new_H = []
        for row in H:
            new_H.append(row + row)
        for row in H:
            new_H.append(row + [-val for val in row])
        H = new_H
    return H


def hadamard_rotate(X, W):
    n = len(X[0])
    H = _hadamard(n)
    scale = 1.0 / math.sqrt(n)
    Q = [[val * scale for val in row] for row in H]

    m = len(X)
    k = len(W[0])

    X_rot = []
    for i in range(m):
        row_res = []
        for j in range(n):
            s = 0.0
            for l in range(n):
                s += X[i][l] * Q[l][j]
            row_res.append(s)
        X_rot.append(row_res)

    W_rot = []
    for i in range(n):
        row_res = []
        for j in range(k):
            s = 0.0
            for l in range(n):
                s += Q[l][i] * W[l][j]
            row_res.append(s)
        W_rot.append(row_res)

    return (X_rot, W_rot)
