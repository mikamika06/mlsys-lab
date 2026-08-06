import math
import numpy as np


def _softmax(x):
    shape = x.shape
    flat_x = x.reshape(-1, shape[-1])
    res_rows = []
    for row in flat_x:
        max_val = row[0]
        for val in row:
            if val > max_val:
                max_val = val
        exps = []
        s = 0.0
        for val in row:
            e = math.exp(val - max_val)
            exps.append(e)
            s += e
        res_row = [e / s for e in exps]
        res_rows.append(res_row)
    return np.array(res_rows, dtype=x.dtype).reshape(shape)


def _matmul(A, B):
    m = A.shape[0]
    k = A.shape[1]
    n = B.shape[1]
    res = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0.0
            for l in range(k):
                s += A[i, l] * B[l, j]
            res[i][j] = s
    return np.array(res, dtype=A.dtype)


def remove_attention_head(Wq, Wk, Wv, Wo, x, head, num_heads):
    d = Wq.shape[1]
    head_dim = d // num_heads
    start = head * head_dim
    end = (head + 1) * head_dim

    Wq_p = np.concatenate([Wq[:, :start], Wq[:, end:]], axis=1)
    Wk_p = np.concatenate([Wk[:, :start], Wk[:, end:]], axis=1)
    Wv_p = np.concatenate([Wv[:, :start], Wv[:, end:]], axis=1)
    Wo_p = np.concatenate([Wo[:start], Wo[end:]], axis=0)

    q = _matmul(x, Wq_p)
    k = _matmul(x, Wk_p)
    v = _matmul(x, Wv_p)

    outputs = []
    scale = math.sqrt(head_dim)
    for i in range(num_heads - 1):
        a = i * head_dim
        b = (i + 1) * head_dim

        n_rows = q.shape[0]
        scores_list = [[0.0] * n_rows for _ in range(n_rows)]
        for r_idx in range(n_rows):
            for c_idx in range(n_rows):
                s = 0.0
                for l in range(head_dim):
                    s += q[r_idx, a + l] * k[c_idx, a + l]
                scores_list[r_idx][c_idx] = s / scale
        scores = np.array(scores_list, dtype=q.dtype)

        probs = _softmax(scores)
        v_slice = v[:, a:b]
        out_dim = v_slice.shape[1]
        out_list = [[0.0] * out_dim for _ in range(n_rows)]
        for r_idx in range(n_rows):
            for c_idx in range(out_dim):
                s = 0.0
                for l in range(n_rows):
                    s += probs[r_idx, l] * v_slice[l, c_idx]
                out_list[r_idx][c_idx] = s
        outputs.append(np.array(out_list, dtype=x.dtype))

    concat = np.concatenate(outputs, axis=1)
    y = _matmul(concat, Wo_p)
    return Wq_p, Wk_p, Wv_p, Wo_p, y
