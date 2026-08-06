import math
import numpy as np


def _split_heads(A, num_heads):
    n, total = A.shape
    d_head = total // num_heads
    res = np.empty((num_heads, n, d_head), dtype=A.dtype)
    for i in range(num_heads):
        for j in range(n):
            for k in range(d_head):
                res[i, j, k] = A[j, i * d_head + k]
    return res


def mla_forward(x, W_Q, W_down_kv, W_up_K, W_up_V, num_heads):
    x = np.asarray(x, dtype=np.float64)
    W_Q = np.asarray(W_Q, dtype=np.float64)
    W_down_kv = np.asarray(W_down_kv, dtype=np.float64)
    W_up_K = np.asarray(W_up_K, dtype=np.float64)
    W_up_V = np.asarray(W_up_V, dtype=np.float64)

    n, d_model = x.shape
    r = W_down_kv.shape[1]
    dim_q = W_Q.shape[1]
    dim_kv = W_up_K.shape[1]

    c_kv = np.empty((n, r), dtype=np.float64)
    for i in range(n):
        for j in range(r):
            acc = 0.0
            for k in range(d_model):
                acc += x[i, k] * W_down_kv[k, j]
            c_kv[i, j] = acc

    Q = np.empty((n, dim_q), dtype=np.float64)
    for i in range(n):
        for j in range(dim_q):
            acc = 0.0
            for k in range(d_model):
                acc += x[i, k] * W_Q[k, j]
            Q[i, j] = acc

    K = np.empty((n, dim_kv), dtype=np.float64)
    for i in range(n):
        for j in range(dim_kv):
            acc = 0.0
            for k in range(r):
                acc += c_kv[i, k] * W_up_K[k, j]
            K[i, j] = acc

    V = np.empty((n, dim_kv), dtype=np.float64)
    for i in range(n):
        for j in range(dim_kv):
            acc = 0.0
            for k in range(r):
                acc += c_kv[i, k] * W_up_V[k, j]
            V[i, j] = acc

    Qh = _split_heads(Q, num_heads)
    Kh = _split_heads(K, num_heads)
    Vh = _split_heads(V, num_heads)
    d_head = Qh.shape[-1]
    scale = 1.0 / math.sqrt(d_head)

    scores = np.empty((num_heads, n, n), dtype=np.float64)
    for h in range(num_heads):
        for i in range(n):
            for j in range(n):
                acc = 0.0
                for k in range(d_head):
                    acc += Qh[h, i, k] * Kh[h, j, k]
                scores[h, i, j] = acc * scale

    for h in range(num_heads):
        for i in range(n):
            max_val = scores[h, i, 0]
            for j in range(1, n):
                if scores[h, i, j] > max_val:
                    max_val = scores[h, i, j]
            for j in range(n):
                scores[h, i, j] = scores[h, i, j] - max_val

    w = np.empty((num_heads, n, n), dtype=np.float64)
    for h in range(num_heads):
        for i in range(n):
            sum_exp = 0.0
            for j in range(n):
                val = math.exp(scores[h, i, j])
                w[h, i, j] = val
                sum_exp += val
            for j in range(n):
                w[h, i, j] = w[h, i, j] / sum_exp

    out_h = np.empty((num_heads, n, d_head), dtype=np.float64)
    for h in range(num_heads):
        for i in range(n):
            for j in range(d_head):
                acc = 0.0
                for k in range(n):
                    acc += w[h, i, k] * Vh[h, k, j]
                out_h[h, i, j] = acc

    out = np.empty((n, num_heads * d_head), dtype=np.float64)
    for h in range(num_heads):
        for i in range(n):
            for k in range(d_head):
                out[i, h * d_head + k] = out_h[h, i, k]

    return out, c_kv
