import math


def streaming_attention(tokens: list[float], q: list[list[float]], k: list[list[float]], v: list[list[float]], S: int, W: int) -> tuple[list[int], list[int], list[list[float]]]:
    T = len(tokens)

    sink_len = min(S, T)
    sink = []
    for i in range(sink_len):
        sink.append(i)

    start = max(0, T - W)
    window_len = T - start
    window = []
    for i in range(window_len):
        window.append(start + i)

    concat = []
    for i in range(sink_len):
        concat.append(sink[i])
    for i in range(window_len):
        concat.append(window[i])

    seen = set()
    unique_list = []
    for val in concat:
        if val not in seen:
            seen.add(val)
            unique_list.append(val)

    idx = unique_list
    idx_len = len(idx)

    pos = []
    for j in range(idx_len):
        i_val = idx[j]
        if i_val < S:
            pos.append(i_val)
        else:
            pos.append(S + i_val - start)

    dim = len(k[0])
    kk = []
    for j in range(idx_len):
        orig_row = idx[j]
        row_vals = []
        for d in range(dim):
            row_vals.append(k[orig_row][d])
        kk.append(row_vals)

    v_dim = len(v[0])
    vv = []
    for j in range(idx_len):
        orig_row = idx[j]
        row_vals = []
        for d in range(v_dim):
            row_vals.append(v[orig_row][d])
        vv.append(row_vals)

    sqrt_dim = math.sqrt(dim)
    q_rows = len(q)
    q_cols = len(q[0])

    logits = []
    for r in range(q_rows):
        row_logits = []
        for c in range(idx_len):
            dot_val = 0.0
            for d in range(q_cols):
                dot_val += q[r][d] * kk[c][d]
            row_logits.append(dot_val / sqrt_dim)
        logits.append(row_logits)

    for r in range(q_rows):
        max_val = logits[r][0]
        for c in range(1, idx_len):
            if logits[r][c] > max_val:
                max_val = logits[r][c]
        for c in range(idx_len):
            logits[r][c] -= max_val

    weights = []
    for r in range(q_rows):
        row_weights = []
        for c in range(idx_len):
            row_weights.append(math.exp(logits[r][c]))
        weights.append(row_weights)

    for r in range(q_rows):
        sum_val = 0.0
        for c in range(idx_len):
            sum_val += weights[r][c]
        for c in range(idx_len):
            weights[r][c] /= sum_val

    out = []
    for r in range(q_rows):
        row_out = []
        for vd in range(v_dim):
            val = 0.0
            for c in range(idx_len):
                val += weights[r][c] * vv[c][vd]
            row_out.append(val)
        out.append(row_out)

    return idx, pos, out
