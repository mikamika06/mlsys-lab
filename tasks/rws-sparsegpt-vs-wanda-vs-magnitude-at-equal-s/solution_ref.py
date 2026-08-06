import math
import numpy as np


def _mse(W, Wh, X):
    m, d = W.shape
    d_x, n = X.shape
    Y = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for k in range(n):
            s = 0.0
            for j in range(d):
                s += W[i, j] * X[j, k]
            Y[i, k] = s
    Yh = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for k in range(n):
            s = 0.0
            for j in range(d):
                s += Wh[i, j] * X[j, k]
            Yh[i, k] = s
    total_sum = 0.0
    count = m * n
    for i in range(m):
        for k in range(n):
            diff = Y[i, k] - Yh[i, k]
            total_sum += diff * diff
    return float(total_sum / count)


def _magnitude(W, sparsity):
    out = np.asarray(W, dtype=np.float64).copy()
    m, n = out.shape
    remove = int(out.size * sparsity)
    flat_list = []
    for i in range(m):
        for j in range(n):
            flat_list.append((abs(out[i, j]), i * n + j))
    sorted_list = sorted(flat_list, key=lambda x: x[0])
    for k in range(remove):
        flat_idx = sorted_list[k][1]
        i, j = divmod(flat_idx, n)
        out[i, j] = 0.0
    return out


def _wanda(W, X, sparsity):
    out = np.asarray(W, dtype=np.float64).copy()
    m, d = out.shape
    d_x, n = X.shape
    z = np.zeros(d, dtype=np.float64)
    for i in range(d):
        s = 0.0
        for j in range(n):
            val = X[i, j]
            s += val * val
        z[i] = math.sqrt(s)
    scores = np.zeros((m, d), dtype=np.float64)
    for i in range(m):
        for j in range(d):
            scores[i, j] = abs(out[i, j]) * z[j]
    remove = int(out.size * sparsity)
    flat_list = []
    for i in range(m):
        for j in range(d):
            flat_list.append((scores[i, j], i * d + j))
    sorted_list = sorted(flat_list, key=lambda x: x[0])
    for k in range(remove):
        flat_idx = sorted_list[k][1]
        i, j = divmod(flat_idx, d)
        out[i, j] = 0.0
    return out


def _sparsegpt(W, X, sparsity, lam):
    W_work = np.asarray(W, dtype=np.float64).copy()
    m, d = W_work.shape
    d_x, n = X.shape
    XT = np.zeros((n, d), dtype=np.float64)
    for i in range(d):
        for j in range(n):
            XT[j, i] = X[i, j]
    XXT = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(d):
            s = 0.0
            for k in range(n):
                s += X[i, k] * XT[k, j]
            XXT[i, j] = s
    H = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(d):
            eye_val = 1.0 if i == j else 0.0
            H[i, j] = 2.0 * XXT[i, j] + lam * eye_val
    L = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(i + 1):
            s = 0.0
            for k in range(j):
                s += L[i, k] * L[j, k]
            if i == j:
                val = H[i, i] - s
                L[i, j] = math.sqrt(val)
            else:
                L[i, j] = (H[i, j] - s) / L[j, j]
    L_inv = np.zeros((d, d), dtype=np.float64)
    for col in range(d):
        for i in range(d):
            if i < col:
                L_inv[i, col] = 0.0
            else:
                s = 0.0
                for k in range(col, i):
                    s += L[i, k] * L_inv[k, col]
                b_val = 1.0 if i == col else 0.0
                L_inv[i, col] = (b_val - s) / L[i, i]
    Hinv = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(d):
            s = 0.0
            for k in range(d):
                s += L_inv[k, i] * L_inv[k, j]
            Hinv[i, j] = s
    remove = int(W_work.size * sparsity)
    diag_Hinv = np.zeros(d, dtype=np.float64)
    for i in range(d):
        diag_Hinv[i] = Hinv[i, i]
    scores = np.zeros((m, d), dtype=np.float64)
    for i in range(m):
        for j in range(d):
            scores[i, j] = (W_work[i, j] * W_work[i, j]) / diag_Hinv[j]
    flat_list = []
    for i in range(m):
        for j in range(d):
            flat_list.append((scores[i, j], i * d + j))
    flat = sorted(flat_list, key=lambda x: x[0])[:remove]
    live = np.ones_like(W_work, dtype=bool)
    for item in flat:
        idx = item[1]
        i, j = divmod(int(idx), d)
        if not live[i, j]:
            continue
        old = W_work[i, j]
        live[i, j] = False
        denom = Hinv[j, j]
        factor = -old / denom
        for col in range(d):
            W_work[i, col] += factor * Hinv[j, col]
        W_work[i, j] = 0.0
    return W_work


def compare_prune_methods_mse(W: np.ndarray, X: np.ndarray, sparsity: float, lam: float) -> dict:
    W_mag = _magnitude(W, sparsity)
    W_wanda = _wanda(W, X, sparsity)
    W_sgpt = _sparsegpt(W, X, sparsity, lam)
    return {
        "mse_magnitude": _mse(W, W_mag, X),
        "mse_wanda": _mse(W, W_wanda, X),
        "mse_sparsegpt": _mse(W, W_sgpt, X),
    }
