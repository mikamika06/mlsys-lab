import numpy as np

def int8_dynamic_act_per_token_x_int8_weight_per_channel(A: np.ndarray,
                                                         W: np.ndarray) -> np.ndarray:
    eps = 1e-12
    m, k = A.shape
    k_w, n = W.shape
    
    scale_row = np.zeros(m, dtype=A.dtype)
    for i in range(m):
        max_val = 0.0
        for j in range(k):
            val = A[i, j]
            if val < 0.0:
                val = -val
            if val > max_val:
                max_val = val
        if max_val < eps:
            max_val = 1.0
        scale_row[i] = max_val

    a_int = np.zeros((m, k), dtype=np.int8)
    for i in range(m):
        sr = scale_row[i]
        for j in range(k):
            val = round(A[i, j] / sr)
            if val < -128.0:
                val = -128.0
            elif val > 127.0:
                val = 127.0
            a_int[i, j] = int(val)

    scale_col = np.zeros(n, dtype=W.dtype)
    for j in range(n):
        max_val = 0.0
        for i in range(k_w):
            val = W[i, j]
            if val < 0.0:
                val = -val
            if val > max_val:
                max_val = val
        if max_val < eps:
            max_val = 1.0
        scale_col[j] = max_val

    w_int = np.zeros((k_w, n), dtype=np.int8)
    for j in range(n):
        sc = scale_col[j]
        for i in range(k_w):
            val = round(W[i, j] / sc)
            if val < -128.0:
                val = -128.0
            elif val > 127.0:
                val = 127.0
            w_int[i, j] = int(val)

    y_int32 = np.zeros((m, n), dtype=np.int32)
    for i in range(m):
        for j in range(n):
            acc = 0
            for p in range(k):
                acc += int(a_int[i, p]) * int(w_int[p, j])
            y_int32[i, j] = acc

    Y = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        sr = scale_row[i]
        for j in range(n):
            sc = scale_col[j]
            Y[i, j] = float(y_int32[i, j]) * (sr * sc)

    return Y
