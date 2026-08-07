import math


def _softmax(matrix):
    res_rows = []
    for row in matrix:
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
    return res_rows


def _matmul(A, B):
    m = len(A)
    k = len(A[0])
    n = len(B[0])
    res = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0.0
            for l in range(k):
                s += A[i][l] * B[l][j]
            res[i][j] = s
    return res


def remove_attention_head(
    Wq: list[list[float]],
    Wk: list[list[float]],
    Wv: list[list[float]],
    Wo: list[list[float]],
    x: list[list[float]],
    head: int,
    num_heads: int,
):
    d = len(Wq[0])
    head_dim = d // num_heads
    start = head * head_dim
    end = (head + 1) * head_dim

    def slice_cols(mat, s_idx, e_idx):
        return [row[:s_idx] + row[e_idx:] for row in mat]

    def slice_rows(mat, s_idx, e_idx):
        return mat[:s_idx] + mat[e_idx:]

    Wq_p = slice_cols(Wq, start, end)
    Wk_p = slice_cols(Wk, start, end)
    Wv_p = slice_cols(Wv, start, end)
    Wo_p = slice_rows(Wo, start, end)

    q = _matmul(x, Wq_p)
    k = _matmul(x, Wk_p)
    v = _matmul(x, Wv_p)

    n_rows = len(q)
    scale = math.sqrt(head_dim)
    outputs = []

    for i in range(num_heads - 1):
        a = i * head_dim
        b = (i + 1) * head_dim

        scores_list = [[0.0] * n_rows for _ in range(n_rows)]
        for r_idx in range(n_rows):
            for c_idx in range(n_rows):
                s = 0.0
                for l in range(head_dim):
                    s += q[r_idx][a + l] * k[c_idx][a + l]
                scores_list[r_idx][c_idx] = s / scale

        probs = _softmax(scores_list)

        v_slice = [row[a:b] for row in v]
        out_dim = len(v_slice[0])

        head_out = [[0.0] * out_dim for _ in range(n_rows)]
        for r_idx in range(n_rows):
            for c_idx in range(out_dim):
                s = 0.0
                for l in range(n_rows):
                    s += probs[r_idx][l] * v_slice[l][c_idx]
                head_out[r_idx][c_idx] = s

        outputs.append(head_out)

    concat = []
    for r_idx in range(n_rows):
        row_concat = []
        for head_out in outputs:
            row_concat.extend(head_out[r_idx])
        concat.append(row_concat)

    y = _matmul(concat, Wo_p)
    return Wq_p, Wk_p, Wv_p, Wo_p, y
