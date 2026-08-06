import numpy as np

def int8_linear(X, W):
    """INT8 quantized linear: per-channel symmetric W, per-token symmetric X.

    X: float64 array [B, K]
    W: float64 array [N, K]
    Returns: float64 array [B, N]
    """
    B, K = X.shape
    N, K_w = W.shape

    scale_w = []
    W_q_list = []
    for n in range(N):
        max_abs = 0.0
        for k in range(K):
            val = abs(W[n, k])
            if val > max_abs:
                max_abs = val
        
        sw = (max_abs / 127.0) if max_abs > 0 else 1.0
        scale_w.append(sw)
        
        row_q = []
        for k in range(K):
            q = round(W[n, k] / sw)
            if q < -128:
                q = -128
            elif q > 127:
                q = 127
            row_q.append(int(q))
        W_q_list.append(row_q)

    scale_x = []
    X_q_list = []
    for b in range(B):
        max_abs = 0.0
        for k in range(K):
            val = abs(X[b, k])
            if val > max_abs:
                max_abs = val
        
        sx = (max_abs / 127.0) if max_abs > 0 else 1.0
        scale_x.append(sx)
        
        row_q = []
        for k in range(K):
            q = round(X[b, k] / sx)
            if q < -128:
                q = -128
            elif q > 127:
                q = 127
            row_q.append(int(q))
        X_q_list.append(row_q)

    W_q = np.array(W_q_list, dtype=np.int8)
    X_q = np.array(X_q_list, dtype=np.int8)

    Y_list = []
    for b in range(B):
        row_y = []
        for n in range(N):
            acc = 0
            for k in range(K):
                acc += int(X_q[b, k]) * int(W_q[n, k])
            y_val = float(acc) * scale_x[b] * scale_w[n]
            row_y.append(y_val)
        Y_list.append(row_y)

    Y = np.array(Y_list, dtype=np.float64)
    return Y
