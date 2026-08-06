import numpy as np


def quantize_classifier_head(X, W, b):
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    C, D = W.shape
    N = X.shape[0]

    scale_list = []
    for i in range(C):
        max_val = 0.0
        for j in range(D):
            val = W[i, j]
            if val < 0.0:
                val = -val
            if val > max_val:
                max_val = val
        s = max_val / 127.0
        if s == 0.0:
            s = 1.0
        scale_list.append(s)
    scale = np.array(scale_list, dtype=np.float64)

    W_int8_list = []
    W_deq_list = []
    for i in range(C):
        s = scale[i]
        row_q = []
        row_deq = []
        for j in range(D):
            val = W[i, j] / s
            rounded = round(val)
            if rounded < -127.0:
                rounded = -127.0
            elif rounded > 127.0:
                rounded = 127.0
            q_val = int(rounded)
            row_q.append(q_val)
            row_deq.append(float(q_val) * s)
        W_int8_list.append(row_q)
        W_deq_list.append(row_deq)

    W_int8 = np.array(W_int8_list, dtype=np.int64)
    W_deq = np.array(W_deq_list, dtype=np.float64)

    logits_list = []
    for i in range(N):
        row_logits = []
        for j in range(C):
            acc = 0.0
            for k in range(D):
                acc += X[i, k] * W_deq[j, k]
            acc += b[j]
            row_logits.append(acc)
        logits_list.append(row_logits)

    logits = np.array(logits_list, dtype=np.float64)

    return logits, W_int8, scale
