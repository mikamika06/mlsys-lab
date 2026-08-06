import math
import numpy as np


def _quantize(a):
    a = np.asarray(a, dtype=np.float64)
    rows, cols = a.shape
    
    max_val = 0.0
    for i in range(rows):
        for j in range(cols):
            v = a[i, j]
            abs_v = -v if v < 0.0 else v
            if abs_v > max_val:
                max_val = abs_v
                
    z = max_val / 127.0
    out = np.zeros_like(a, dtype=np.float64)
    if z == 0.0:
        return out

    for i in range(rows):
        for j in range(cols):
            val = a[i, j] / z
            rounded = math.floor(val + 0.5) if val >= 0.0 else math.ceil(val - 0.5)
            if rounded < -127.0:
                clipped = -127.0
            elif rounded > 127.0:
                clipped = 127.0
            else:
                clipped = rounded
            out[i, j] = clipped * z
            
    return out


def awq_grid_scale(W, X, steps=41):
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    M, N = W.shape
    _, K = X.shape

    col = [0.0] * N
    for j in range(N):
        max_col = 0.0
        for i in range(M):
            v = W[i, j]
            abs_v = -v if v < 0.0 else v
            if abs_v > max_col:
                max_col = abs_v
        col[j] = max_col + 1e-8

    WX = np.zeros((M, K), dtype=np.float64)
    for i in range(M):
        for k in range(K):
            acc = 0.0
            for j in range(N):
                acc += W[i, j] * X[j, k]
            WX[i, k] = acc

    best = None
    best_mse = float("inf")

    alphas = [0.0 + i * (1.0 - 0.0) / (steps - 1) for i in range(steps)] if steps > 1 else [0.0]

    scaled_W = np.zeros((M, N), dtype=np.float64)
    unscaled_q = np.zeros((M, N), dtype=np.float64)
    out_mat = np.zeros((M, K), dtype=np.float64)

    for alpha in alphas:
        s = [col[j] ** alpha for j in range(N)]

        for i in range(M):
            for j in range(N):
                scaled_W[i, j] = W[i, j] * s[j]

        q = _quantize(scaled_W)

        for i in range(M):
            for j in range(N):
                unscaled_q[i, j] = q[i, j] / s[j]

        for i in range(M):
            for k in range(K):
                acc = 0.0
                for j in range(N):
                    acc += unscaled_q[i, j] * X[j, k]
                out_mat[i, k] = acc

        total_sq_diff = 0.0
        for i in range(M):
            for k in range(K):
                diff = WX[i, k] - out_mat[i, k]
                total_sq_diff += diff * diff

        mse = total_sq_diff / (M * K)

        if mse < best_mse:
            best_mse = mse
            best = list(s)

    return np.asarray(best, dtype=np.float64)
