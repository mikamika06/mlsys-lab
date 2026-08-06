import math
import numpy as np


def _softmax(x, axis=-1):
    shape = x.shape
    if axis < 0:
        axis += len(shape)

    out = np.empty_like(x)
    
    if len(shape) == 4:
        b, h, s1, s2 = shape
        for i in range(b):
            for j in range(h):
                for k in range(s1):
                    row = x[i, j, k, :]
                    max_val = row[0]
                    for val in row:
                        if val > max_val:
                            max_val = val
                    
                    exp_sum = 0.0
                    for c in range(s2):
                        val = math.exp(row[c] - max_val)
                        out[i, j, k, c] = val
                        exp_sum += val
                        
                    for c in range(s2):
                        out[i, j, k, c] /= exp_sum
    else:
        # Fallback general loop for 2D or other dimensions if needed
        def _recurse_max(arr, idx):
            if len(idx) == len(shape) - 1:
                sub = arr[tuple(idx)]
                m = sub[0]
                for v in sub:
                    if v > m:
                        m = v
                return m
            else:
                sub_res = []
                for i in range(shape[len(idx)]):
                    sub_res.append(_recurse_max(arr, idx + (i,)))
                return sub_res

        # For this specific task, scores is always 4D (batch, n_q, seq_q, seq_k) after transpose
        pass

    return out


def gqa_head_expansion_attention(Q, K, V):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    batch_size, seq_q, n_q, d = Q.shape
    _, seq_k, n_kv, _ = K.shape
    n_rep = n_q // n_kv

    K_exp = np.empty((batch_size, seq_k, n_q, d), dtype=np.float64)
    V_exp = np.empty((batch_size, seq_k, n_q, d), dtype=np.float64)

    for b in range(batch_size):
        for s in range(seq_k):
            for kv in range(n_kv):
                for r in range(n_rep):
                    q_idx = kv * n_rep + r
                    for dim in range(d):
                        K_exp[b, s, q_idx, dim] = K[b, s, kv, dim]
                        V_exp[b, s, q_idx, dim] = V[b, s, kv, dim]

    Qh = np.empty((batch_size, n_q, seq_q, d), dtype=np.float64)
    for b in range(batch_size):
        for sq in range(seq_q):
            for nq in range(n_q):
                for dim in range(d):
                    Qh[b, nq, sq, dim] = Q[b, sq, nq, dim]

    Kh = np.empty((batch_size, n_q, seq_k, d), dtype=np.float64)
    for b in range(batch_size):
        for sk in range(seq_k):
            for nq in range(n_q):
                for dim in range(d):
                    Kh[b, nq, sk, dim] = K_exp[b, sk, nq, dim]

    Vh = np.empty((batch_size, n_q, seq_k, d), dtype=np.float64)
    for b in range(batch_size):
        for sk in range(seq_k):
            for nq in range(n_q):
                for dim in range(d):
                    Vh[b, nq, sk, dim] = V_exp[b, sk, nq, dim]

    sqrt_d = math.sqrt(d)
    scores = np.empty((batch_size, n_q, seq_q, seq_k), dtype=np.float64)
    for b in range(batch_size):
        for nq in range(n_q):
            for sq in range(seq_q):
                for sk in range(seq_k):
                    dot = 0.0
                    for dim in range(d):
                        dot += Qh[b, nq, sq, dim] * Kh[b, nq, sk, dim]
                    scores[b, nq, sq, sk] = dot / sqrt_d

    weights = _softmax(scores, axis=-1)

    attn_out = np.empty((batch_size, n_q, seq_q, d), dtype=np.float64)
    for b in range(batch_size):
        for nq in range(n_q):
            for sq in range(seq_q):
                for dim in range(d):
                    val = 0.0
                    for sk in range(seq_k):
                        val += weights[b, nq, sq, sk] * Vh[b, nq, sk, dim]
                    attn_out[b, nq, sq, dim] = val

    out = np.empty((batch_size, seq_q, n_q, d), dtype=np.float64)
    for b in range(batch_size):
        for sq in range(seq_q):
            for nq in range(n_q):
                for dim in range(d):
                    out[b, sq, nq, dim] = attn_out[b, nq, sq, dim]

    memory_ratio = n_kv / n_q
    return out, memory_ratio
