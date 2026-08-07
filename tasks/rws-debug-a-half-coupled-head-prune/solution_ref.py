import math


def matmul(a, b):
    rows_a = len(a)
    cols_a = len(a[0])
    rows_b = len(b)
    cols_b = len(b[0])

    result = [[0.0 for _ in range(cols_b)] for _ in range(rows_a)]
    for i in range(rows_a):
        for k in range(cols_a):
            aik = a[i][k]
            for j in range(cols_b):
                result[i][j] += aik * b[k][j]
    return result


def transpose(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    return [[matrix[r][c] for r in range(rows)] for c in range(cols)]


def pruned_attention_forward(x: list[list[float]], q_proj: list[list[float]], k_proj: list[list[float]], v_proj: list[list[float]], o_proj: list[list[float]], heads: int, keep_heads: list[int]) -> list[list[float]]:
    n = len(x)
    d = len(x[0])
    head_dim = d // heads

    q_full = matmul(x, q_proj)
    k_full = matmul(x, k_proj)
    v_full = matmul(x, v_proj)

    q = []
    k = []
    v = []
    for i in range(n):
        row_q = []
        row_k = []
        row_v = []
        for h in range(heads):
            start = h * head_dim
            end = (h + 1) * head_dim
            row_q.append(q_full[i][start:end])
            row_k.append(k_full[i][start:end])
            row_v.append(v_full[i][start:end])
        q.append(row_q)
        k.append(row_k)
        v.append(row_v)

    outputs = []
    scale = math.sqrt(head_dim)

    for h in keep_heads:
        q_h = [[q[i][h][j] for j in range(head_dim)] for i in range(n)]
        k_h = [[k[i][h][j] for j in range(head_dim)] for i in range(n)]
        v_h = [[v[i][h][j] for j in range(head_dim)] for i in range(n)]
        k_h_t = transpose(k_h)

        scores = matmul(q_h, k_h_t)
        for i in range(n):
            for j in range(n):
                scores[i][j] /= scale

        probs = []
        for i in range(n):
            row = scores[i]
            max_val = max(row)
            exps = [math.exp(val - max_val) for val in row]
            sum_exps = sum(exps)
            probs.append([e / sum_exps for e in exps])

        outputs.append(matmul(probs, v_h))

    z = []
    for i in range(n):
        row_z = []
        for h in keep_heads:
            idx = keep_heads.index(h)
            row_z.extend(outputs[idx][i])
        z.append(row_z)

    cols = []
    for h in keep_heads:
        cols.extend(range(h * head_dim, (h + 1) * head_dim))

    o_proj_sub = [[o_proj[c][col] for col in range(len(o_proj[0]))] for c in cols]
    return matmul(z, o_proj_sub)
