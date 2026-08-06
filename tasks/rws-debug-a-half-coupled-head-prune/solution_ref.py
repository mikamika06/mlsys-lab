import math
import numpy as np


def pruned_attention_forward(x, q_proj, k_proj, v_proj, o_proj, heads, keep_heads):
    n, d = x.shape
    head_dim = d // heads

    q_full = [[sum(x[i][k] * q_proj[k][j] for k in range(d)) for j in range(d)] for i in range(n)]
    k_full = [[sum(x[i][k] * k_proj[k][j] for k in range(d)) for j in range(d)] for i in range(n)]
    v_full = [[sum(x[i][k] * v_proj[k][j] for k in range(d)) for j in range(d)] for i in range(n)]

    q = [[[q_full[i][h * head_dim + jj] for jj in range(head_dim)] for h in range(heads)] for i in range(n)]
    k = [[[k_full[i][h * head_dim + jj] for jj in range(head_dim)] for h in range(heads)] for i in range(n)]
    v = [[[v_full[i][h * head_dim + jj] for jj in range(head_dim)] for h in range(heads)] for i in range(n)]

    scale = math.sqrt(head_dim)
    outputs = []
    for h in keep_heads:
        scores = []
        for i in range(n):
            row = []
            for j in range(n):
                dot = sum(q[i][h][c] * k[j][h][c] for c in range(head_dim))
                row.append(dot / scale)
            scores.append(row)

        max_scores = []
        for i in range(n):
            m = scores[i][0]
            for j in range(1, n):
                if scores[i][j] > m:
                    m = scores[i][j]
            max_scores.append(m)

        probs = []
        for i in range(n):
            row = []
            s = 0.0
            for j in range(n):
                val = math.exp(scores[i][j] - max_scores[i])
                row.append(val)
                s += val
            row_probs = [val / s for val in row]
            probs.append(row_probs)

        head_out = []
        for i in range(n):
            row = []
            for c in range(head_dim):
                val = sum(probs[i][j] * v[j][h][c] for j in range(n))
                row.append(val)
            head_out.append(row)
        outputs.append(head_out)

    z = []
    for i in range(n):
        row = []
        for head_out in outputs:
            row.extend(head_out[i])
        z.append(row)

    cols = []
    for h in keep_heads:
        cols.extend(range(h * head_dim, (h + 1) * head_dim))

    z_arr = np.array(z, dtype=x.dtype)
    o_sub = o_proj[cols, :]

    result = [[sum(z_arr[i][k] * o_sub[k][j] for k in range(len(cols))) for j in range(d)] for i in range(n)]
    return np.array(result, dtype=x.dtype)
