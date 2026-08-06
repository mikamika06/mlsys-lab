import math
import numpy as np

def leading_eigenvector(X, num_iter=1000):
    n, d = X.shape
    means = [0.0] * d
    for j in range(d):
        s = 0.0
        for i in range(n):
            s += float(X[i, j])
        means[j] = s / n

    Xc = [[0.0] * d for _ in range(n)]
    for i in range(n):
        for j in range(d):
            Xc[i][j] = float(X[i, j]) - means[j]

    v = [1.0] * d
    s = 0.0
    for j in range(d):
        s += v[j] * v[j]
    norm_v = math.sqrt(s)
    for j in range(d):
        v[j] /= norm_v

    for _ in range(num_iter):
        y = [0.0] * n
        for i in range(n):
            s = 0.0
            for j in range(d):
                s += Xc[i][j] * v[j]
            y[i] = s

        w = [0.0] * d
        for j in range(d):
            s = 0.0
            for i in range(n):
                s += Xc[i][j] * y[i]
            w[j] = s

        norm_sq = 0.0
        for j in range(d):
            norm_sq += w[j] * w[j]
        norm = math.sqrt(norm_sq)

        v = w
        if norm == 0:
            break
        for j in range(d):
            v[j] /= norm

    return np.array(v, dtype=np.float64)
