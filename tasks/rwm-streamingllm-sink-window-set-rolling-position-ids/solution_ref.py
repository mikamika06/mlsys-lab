import math
import numpy as np


def streaming_attention(tokens, q, k, v, S, W):
    T = len(tokens)
    
    sink_len = min(S, T)
    sink = np.empty(sink_len, dtype=np.int64)
    for i in range(sink_len):
        sink[i] = i

    start = max(0, T - W)
    window_len = T - start
    window = np.empty(window_len, dtype=np.int64)
    for i in range(window_len):
        window[i] = start + i

    concat_len = sink_len + window_len
    concat = np.empty(concat_len, dtype=np.int64)
    for i in range(sink_len):
        concat[i] = sink[i]
    for i in range(window_len):
        concat[sink_len + i] = window[i]

    seen = set()
    unique_list = []
    for val in concat:
        if val not in seen:
            seen.add(val)
            unique_list.append(val)

    idx = np.array(unique_list, dtype=np.int64)
    idx_len = len(idx)

    pos = np.empty(idx_len, dtype=np.int64)
    for j in range(idx_len):
        i_val = idx[j]
        if i_val < S:
            pos[j] = i_val
        else:
            pos[j] = S + i_val - start

    dim = k.shape[1]
    kk = np.empty((idx_len, dim), dtype=k.dtype)
    for j in range(idx_len):
        orig_row = idx[j]
        for d in range(dim):
            kk[j, d] = k[orig_row, d]

    v_dim = v.shape[1]
    vv = np.empty((idx_len, v_dim), dtype=v.dtype)
    for j in range(idx_len):
        orig_row = idx[j]
        for d in range(v_dim):
            vv[j, d] = v[orig_row, d]

    sqrt_dim = math.sqrt(dim)
    q_rows = q.shape[0]
    q_cols = q.shape[1]

    logits = np.empty((q_rows, idx_len), dtype=np.float64)
    for r in range(q_rows):
        for c in range(idx_len):
            dot_val = 0.0
            for d in range(q_cols):
                dot_val += q[r, d] * kk[c, d]
            logits[r, c] = dot_val / sqrt_dim

    for r in range(q_rows):
        max_val = logits[r, 0]
        for c in range(1, idx_len):
            if logits[r, c] > max_val:
                max_val = logits[r, c]
        for c in range(idx_len):
            logits[r, c] -= max_val

    weights = np.empty((q_rows, idx_len), dtype=np.float64)
    for r in range(q_rows):
        for c in range(idx_len):
            weights[r, c] = math.exp(logits[r, c])

    for r in range(q_rows):
        sum_val = 0.0
        for c in range(idx_len):
            sum_val += weights[r, c]
        for c in range(idx_len):
            weights[r, c] /= sum_val

    out = np.empty((q_rows, v_dim), dtype=np.float64)
    for r in range(q_rows):
        for vd in range(v_dim):
            val = 0.0
            for c in range(idx_len):
                val += weights[r, c] * vv[c, vd]
            out[r, vd] = val

    return idx, pos, out
