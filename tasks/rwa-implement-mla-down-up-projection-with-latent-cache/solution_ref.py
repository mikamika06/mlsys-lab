import numpy as np


def _split_heads(A, num_heads):
    n, total = A.shape
    d_head = total // num_heads
    return A.reshape(n, num_heads, d_head).transpose(1, 0, 2)


def mla_forward(x, W_Q, W_down_kv, W_up_K, W_up_V, num_heads):
    x = np.asarray(x, dtype=np.float64)
    W_Q = np.asarray(W_Q, dtype=np.float64)
    W_down_kv = np.asarray(W_down_kv, dtype=np.float64)
    W_up_K = np.asarray(W_up_K, dtype=np.float64)
    W_up_V = np.asarray(W_up_V, dtype=np.float64)

    n = x.shape[0]

    c_kv = x @ W_down_kv

    Q = x @ W_Q
    K = c_kv @ W_up_K
    V = c_kv @ W_up_V

    Qh = _split_heads(Q, num_heads)
    Kh = _split_heads(K, num_heads)
    Vh = _split_heads(V, num_heads)
    d_head = Qh.shape[-1]
    scale = 1.0 / np.sqrt(d_head)

    scores = np.matmul(Qh, Kh.transpose(0, 2, 1)) * scale
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / np.sum(w, axis=-1, keepdims=True)
    out_h = np.matmul(w, Vh)

    out = out_h.transpose(1, 0, 2).reshape(n, num_heads * d_head)
    return out, c_kv
