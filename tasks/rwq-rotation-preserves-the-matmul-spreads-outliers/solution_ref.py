import math
import numpy as np


def _hadamard(n):
    h = np.array([[1.0]], dtype=np.float64)
    while h.shape[0] < n:
        h11 = h
        h12 = h
        h21 = h
        h22 = -h
        rows_top = []
        for r in range(h11.shape[0]):
            row = []
            for c in range(h11.shape[1]):
                row.append(h11[r, c])
            for c in range(h12.shape[1]):
                row.append(h12[r, c])
            rows_top.append(row)
        rows_bot = []
        for r in range(h21.shape[0]):
            row = []
            for c in range(h21.shape[1]):
                row.append(h21[r, c])
            for c in range(h22.shape[1]):
                row.append(h22[r, c])
            rows_bot.append(row)
        h = np.array(rows_top + rows_bot, dtype=np.float64)
    
    scale = math.sqrt(float(n))
    res = np.empty_like(h)
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            res[i, j] = h[i, j] / scale
    return res


def _quantize_int4(x):
    """Symmetric per-tensor absmax int4 round-trip (quant then immediate dequant)."""
    x = np.asarray(x, dtype=np.float64)
    qmax = 2 ** (4 - 1) - 1  # 7
    
    max_val = 0.0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            val = x[i, j]
            if val < 0.0:
                val = -val
            if val > max_val:
                max_val = val
    scale = float(max_val)
    scale = scale / qmax if scale > 0 else 1.0
    
    code = np.empty_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            v = x[i, j] / scale
            if v >= 0.0:
                r = math.floor(v + 0.5)
            else:
                r = math.ceil(v - 0.5)
            if r > qmax:
                r = qmax
            elif r < -qmax:
                r = -qmax
            code[i, j] = r
            
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out[i, j] = code[i, j] * scale
    return out


def rotate_and_quantize_matmul(X, W):
    """QuaRot-style rotation invariance and its quantization payoff."""
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    d = X.shape[1]
    H = _hadamard(d)

    n_rows_X = X.shape[0]
    n_cols_W = W.shape[1]
    
    ref = np.zeros((n_rows_X, n_cols_W), dtype=np.float64)
    for i in range(n_rows_X):
        for j in range(n_cols_W):
            acc = 0.0
            for k in range(d):
                acc += X[i, k] * W[k, j]
            ref[i, j] = acc

    Ht = np.empty_like(H)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            Ht[i, j] = H[j, i]

    Xr = np.zeros((n_rows_X, d), dtype=np.float64)
    for i in range(n_rows_X):
        for j in range(d):
            acc = 0.0
            for k in range(d):
                acc += X[i, k] * Ht[k, j]
            Xr[i, j] = acc

    Wr = np.zeros((d, n_cols_W), dtype=np.float64)
    for i in range(d):
        for j in range(n_cols_W):
            acc = 0.0
            for k in range(d):
                acc += H[i, k] * W[k, j]
            Wr[i, j] = acc

    out_rotated = np.zeros((n_rows_X, n_cols_W), dtype=np.float64)
    for i in range(n_rows_X):
        for j in range(n_cols_W):
            acc = 0.0
            for k in range(d):
                acc += Xr[i, k] * Wr[k, j]
            out_rotated[i, j] = acc

    Xq = _quantize_int4(X)
    Wq = _quantize_int4(W)
    
    Xq_Wq = np.zeros((n_rows_X, n_cols_W), dtype=np.float64)
    for i in range(n_rows_X):
        for j in range(n_cols_W):
            acc = 0.0
            for k in range(d):
                acc += Xq[i, k] * Wq[k, j]
            Xq_Wq[i, j] = acc

    mse_unrotated_acc = 0.0
    count_unrotated = 0
    for i in range(n_rows_X):
        for j in range(n_cols_W):
            diff = ref[i, j] - Xq_Wq[i, j]
            mse_unrotated_acc += diff * diff
            count_unrotated += 1
    mse_unrotated = float(mse_unrotated_acc / count_unrotated)

    Xrq = _quantize_int4(Xr)
    Wrq = _quantize_int4(Wr)
    
    Xrq_Wrq = np.zeros((n_rows_X, n_cols_W), dtype=np.float64)
    for i in range(n_rows_X):
        for j in range(n_cols_W):
            acc = 0.0
            for k in range(d):
                acc += Xrq[i, k] * Wrq[k, j]
            Xrq_Wrq[i, j] = acc

    mse_rotated_acc = 0.0
    count_rotated = 0
    for i in range(n_rows_X):
        for j in range(n_cols_W):
            diff = ref[i, j] - Xrq_Wrq[i, j]
            mse_rotated_acc += diff * diff
            count_rotated += 1
    mse_rotated = float(mse_rotated_acc / count_rotated)

    return out_rotated, mse_unrotated, mse_rotated
