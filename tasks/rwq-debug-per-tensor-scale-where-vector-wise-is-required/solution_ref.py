import numpy as np


def _quantize_rows(A):
    A = np.asarray(A, dtype=np.float64)
    M, N = A.shape
    scales_list = []
    for i in range(M):
        max_abs = 0.0
        for j in range(N):
            val = abs(A[i, j])
            if val > max_abs:
                max_abs = val
        scale = max_abs / 127.0
        if scale == 0.0:
            scale = 1.0
        scales_list.append(scale)
    scales = np.array(scales_list, dtype=np.float64)

    q_list = []
    for i in range(M):
        row_q = []
        scale = scales[i]
        for j in range(N):
            v = round(A[i, j] / scale)
            if v < -127:
                v = -127
            elif v > 127:
                v = 127
            row_q.append(int(v))
        q_list.append(row_q)
    q = np.array(q_list, dtype=np.int8)
    return q, scales


def _quantize_cols(A):
    A = np.asarray(A, dtype=np.float64)
    R, C = A.shape
    scales_list = []
    for j in range(C):
        max_abs = 0.0
        for i in range(R):
            val = abs(A[i, j])
            if val > max_abs:
                max_abs = val
        scale = max_abs / 127.0
        if scale == 0.0:
            scale = 1.0
        scales_list.append(scale)
    scales = np.array(scales_list, dtype=np.float64)

    q_list = []
    for i in range(R):
        row_q = []
        for j in range(C):
            scale = scales[j]
            v = round(A[i, j] / scale)
            if v < -127:
                v = -127
            elif v > 127:
                v = 127
            row_q.append(int(v))
        q_list.append(row_q)
    q = np.array(q_list, dtype=np.int8)
    return q, scales


def int8_matmul_per_channel(X, W):
    xq, xs = _quantize_rows(X)
    wq, ws = _quantize_cols(W)
    M = xq.shape[0]
    K = xq.shape[1]
    N = wq.shape[1]

    acc_list = []
    for i in range(M):
        row_acc = []
        for j in range(N):
            s = 0
            for k in range(K):
                s += int(xq[i, k]) * int(wq[k, j])
            row_acc.append(int(s))
        acc_list.append(row_acc)
    acc = np.array(acc_list, dtype=np.int32)

    res_list = []
    for i in range(M):
        row_res = []
        for j in range(N):
            val = float(acc[i, j]) * xs[i] * ws[j]
            row_res.append(val)
        res_list.append(row_res)
    return np.array(res_list, dtype=np.float64)
