import math


def _causal_attention(q, k, v):
    d = len(q[0])
    n = len(q)

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
                    dot += q[i][l] * k[j][l]
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

    v_dim = len(v[0])
    result = []
    for i in range(n):
        row_res = []
        for l in range(v_dim):
            acc = 0.0
            for j in range(n):
                acc += w_rows[i][j] * v[j][l]
            row_res.append(acc)
        result.append(row_res)

    return result


def cow_prefix_attention(q_a, k_a, v_a, q_b, k_b, v_b, shared_prefix_len, block_size):
    len_a = len(q_a)
    len_b = len(q_b)

    blocks_a = -(-len_a // block_size)
    blocks_b = -(-len_b // block_size)
    shared_blocks = min(shared_prefix_len // block_size, blocks_a, blocks_b)

    duplicated = blocks_a + blocks_b
    unique = duplicated - shared_blocks
    size_ratio = float(duplicated / unique) if unique else 0.0

    out_a = _causal_attention(q_a, k_a, v_a)
    out_b = _causal_attention(q_b, k_b, v_b)

    return size_ratio, out_a, out_b
