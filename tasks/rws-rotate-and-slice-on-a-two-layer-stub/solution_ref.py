import math
import numpy as np


def rotate_and_slice(W1, b1, W2, b2, X_cal, X, k):
    n_cal = X_cal.shape[0]
    d = X_cal.shape[1]
    hidden = W1.shape[1]

    H_cal = np.zeros((n_cal, hidden), dtype=X_cal.dtype)
    for i in range(n_cal):
        for j in range(hidden):
            acc = 0.0
            for c in range(d):
                acc += X_cal[i, c] * W1[c, j]
            H_cal[i, j] = acc + b1[j]

    mean_H = np.zeros((1, hidden), dtype=H_cal.dtype)
    for j in range(hidden):
        acc = 0.0
        for i in range(n_cal):
            acc += H_cal[i, j]
        mean_H[0, j] = acc / n_cal

    centered = np.zeros((n_cal, hidden), dtype=H_cal.dtype)
    for i in range(n_cal):
        for j in range(hidden):
            centered[i, j] = H_cal[i, j] - mean_H[0, j]

    denom = n_cal - 1
    cov = np.zeros((hidden, hidden), dtype=centered.dtype)
    for i in range(hidden):
        for j in range(hidden):
            acc = 0.0
            for c in range(n_cal):
                acc += centered[c, i] * centered[c, j]
            cov[i, j] = acc / denom

    n = hidden
    V = np.eye(n, dtype=cov.dtype)
    D = cov.copy()

    for _ in range(200):
        max_val = 0.0
        p, q = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                val = D[i, j]
                if val < 0:
                    val = -val
                if val > max_val:
                    max_val = val
                    p, q = i, j
        if max_val < 1e-14:
            break

        diff = D[q, q] - D[p, p]
        if math.fabs(diff) + max_val == math.fabs(diff):
            t = D[p, q] / diff
        else:
            theta = 0.5 * math.atan2(2.0 * D[p, q], diff)
            t = math.tan(theta)

        c_val = 1.0 / math.sqrt(1.0 + t * t)
        s_val = t * c_val

        tau = s_val / (1.0 + c_val)
        temp = D[p, q]
        D[p, q] = 0.0
        D[p, p] -= t * temp
        D[q, q] += t * temp

        for i in range(p):
            g = D[i, p]
            h = D[i, q]
            D[i, p] = g - s_val * (h + g * tau)
            D[i, q] = h + s_val * (g - h * tau)
        for i in range(p + 1, q):
            g = D[p, i]
            h = D[i, q]
            D[p, i] = g - s_val * (h + g * tau)
            D[i, q] = h + s_val * (g - h * tau)
        for i in range(q + 1, n):
            g = D[p, i]
            h = D[q, i]
            D[p, i] = g - s_val * (h + g * tau)
            D[q, i] = h + s_val * (g - h * tau)

        for i in range(n):
            g = V[i, p]
            h = V[i, q]
            V[i, p] = g - s_val * (h + g * tau)
            V[i, q] = h + s_val * (g - h * tau)

    eigenvalues = np.array([D[i, i] for i in range(n)], dtype=cov.dtype)
    eigenvectors = V

    indexed_evals = []
    for i in range(n):
        indexed_evals.append((eigenvalues[i], i))

    sorted_indexed = sorted(indexed_evals, key=lambda x: x[0], reverse=True)
    order = [item[1] for item in sorted_indexed]

    Q = np.zeros((hidden, hidden), dtype=eigenvectors.dtype)
    for j in range(hidden):
        src_col = order[j]
        for i in range(hidden):
            Q[i, j] = eigenvectors[i, src_col]

    n_X = X.shape[0]
    H = np.zeros((n_X, hidden), dtype=X.dtype)
    for i in range(n_X):
        for j in range(hidden):
            acc = 0.0
            for c in range(d):
                acc += X[i, c] * W1[c, j]
            H[i, j] = acc + b1[j]

    out_dim = W2.shape[1]
    W2_rot = np.zeros((hidden, out_dim), dtype=W2.dtype)
    for i in range(hidden):
        for j in range(out_dim):
            acc = 0.0
            for c in range(hidden):
                acc += Q[c, i] * W2[c, j]
            W2_rot[i, j] = acc

    H_Q_k = np.zeros((n_X, k), dtype=H.dtype)
    for i in range(n_X):
        for j in range(k):
            acc = 0.0
            for m in range(hidden):
                acc += H[i, m] * Q[m, j]
            H_Q_k[i, j] = acc

    result = np.zeros((n_X, out_dim), dtype=H_Q_k.dtype)
    for i in range(n_X):
        for j in range(out_dim):
            acc = 0.0
            for m in range(k):
                acc += H_Q_k[i, m] * W2_rot[m, j]
            result[i, j] = acc + b2[j]

    return result
