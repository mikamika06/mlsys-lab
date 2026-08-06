import math
import numpy as np

def compare_awq_rtn(W: np.ndarray, X: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    rows_W, cols_W = W.shape
    rows_X, cols_X = X.shape

    def quantize(M: np.ndarray) -> np.ndarray:
        rows, cols = M.shape
        out = np.empty((rows, cols), dtype=M.dtype)
        for i in range(rows):
            abs_max = 0.0
            for j in range(cols):
                val = M[i, j]
                v_abs = val if val >= 0 else -val
                if v_abs > abs_max:
                    abs_max = v_abs
            delta = abs_max / 7.0
            if delta == 0.0:
                delta = 1e-9
            for j in range(cols):
                val = M[i, j]
                q = round(val / delta)
                if q < -8.0:
                    q = -8.0
                elif q > 7.0:
                    q = 7.0
                out[i, j] = q * delta
        return out

    Y_true = np.empty((rows_X, rows_W), dtype=float)
    for i in range(rows_X):
        for j in range(rows_W):
            acc = 0.0
            for k in range(cols_X):
                acc += float(X[i, k]) * float(W[j, k])
            Y_true[i, j] = acc

    W_rtn_q = quantize(W)
    Y_rtn = np.empty((rows_X, rows_W), dtype=float)
    for i in range(rows_X):
        for j in range(rows_W):
            acc = 0.0
            for k in range(cols_X):
                acc += float(X[i, k]) * float(W_rtn_q[j, k])
            Y_rtn[i, j] = acc

    diff_rtn_sq = 0.0
    true_sq = 0.0
    for i in range(rows_X):
        for j in range(rows_W):
            diff = Y_rtn[i, j] - Y_true[i, j]
            diff_rtn_sq += diff * diff
            true_val = Y_true[i, j]
            true_sq += true_val * true_val

    err_rtn = float(math.sqrt(diff_rtn_sq) / math.sqrt(true_sq))

    W_scaled = np.empty((rows_W, cols_W), dtype=float)
    for i in range(rows_W):
        for j in range(cols_W):
            W_scaled[i, j] = float(W[i, j]) * float(s[j])

    W_scaled_q = quantize(W_scaled)

    W_awq = np.empty((rows_W, cols_W), dtype=float)
    for i in range(rows_W):
        for j in range(cols_W):
            W_awq[i, j] = W_scaled_q[i, j] / float(s[j])

    Y_awq = np.empty((rows_X, rows_W), dtype=float)
    for i in range(rows_X):
        for j in range(rows_W):
            acc = 0.0
            for k in range(cols_X):
                acc += float(X[i, k]) * float(W_awq[j, k])
            Y_awq[i, j] = acc

    diff_awq_sq = 0.0
    for i in range(rows_X):
        for j in range(rows_W):
            diff = Y_awq[i, j] - Y_true[i, j]
            diff_awq_sq += diff * diff

    err_awq = float(math.sqrt(diff_awq_sq) / math.sqrt(true_sq))

    return err_rtn, err_awq
