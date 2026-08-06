import math
import numpy as np


def _causal_attention(q, k, v):
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    d = q.shape[-1]
    n = q.shape[0]

    sqrt_d = math.sqrt(d)
    scores = []
    for i in range(n):
        row = []
        for j in range(n):
            if j > i:
                row.append(-float("inf"))
            else:
                dot = 0.0
                for l in range(d):
                    dot += q[i, l] * k[j, l]
                row.append(dot / sqrt_d)
        scores.append(row)

    shifted = []
    for i in range(n):
        max_val = -float("inf")
        for val in scores[i]:
            if val > max_val:
                max_val = val
        row_shifted = []
        for val in scores[i]:
            row_shifted.append(val - max_val)
        shifted.append(row_shifted)

    exp_scores = []
    for i in range(n):
        row_exp = []
        for val in shifted[i]:
            row_exp.append(math.exp(val))
        exp_scores.append(row_exp)

    w_rows = []
    for i in range(n):
        s_val = 0.0
        for val in exp_scores[i]:
            s_val += val
        row_w = []
        for val in exp_scores[i]:
            row_w.append(val / s_val)
        w_rows.append(row_w)

    v_dim = v.shape[-1]
    result = []
    for i in range(n):
        row_res = []
        for l in range(v_dim):
            acc = 0.0
            for j in range(n):
                acc += w_rows[i][j] * v[j, l]
            row_res.append(acc)
        result.append(row_res)

    return np.array(result, dtype=np.float64)


def cow_prefix_attention(q_a, k_a, v_a, q_b, k_b, v_b, shared_prefix_len, block_size):
    len_a = np.asarray(q_a).shape[0]
    len_b = np.asarray(q_b).shape[0]

    blocks_a = -(-len_a // block_size)
    blocks_b = -(-len_b // block_size)
    shared_blocks = min(shared_prefix_len // block_size, blocks_a, blocks_b)

    duplicated = blocks_a + blocks_b
    unique = duplicated - shared_blocks
    size_ratio = float(duplicated / unique) if unique else 0.0

    out_a = _causal_attention(q_a, k_a, v_a)
    out_b = _causal_attention(q_b, k_b, v_b)

    return size_ratio, out_a, out_b
