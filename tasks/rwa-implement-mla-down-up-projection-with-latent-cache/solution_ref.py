import math


def _split_heads(
    A: list[list[float]], num_heads: int
) -> list[list[list[float]]]:
    n = len(A)
    total = len(A[0])
    d_head = total // num_heads
    res = [[[0.0] * d_head for _ in range(n)] for _ in range(num_heads)]
    for i in range(num_heads):
        for j in range(n):
            for k in range(d_head):
                res[i][j][k] = A[j][i * d_head + k]
    return res


def mla_forward(
    x: list[list[float]],
    W_Q: list[list[float]],
    W_down_kv: list[list[float]],
    W_up_K: list[list[float]],
    W_up_V: list[list[float]],
    num_heads: int,
) -> tuple[list[list[float]], list[list[float]]]:
    n = len(x)
    d_model = len(x[0])
    r = len(W_down_kv[0])
    dim_q = len(W_Q[0])
    dim_kv = len(W_up_K[0])

    c_kv = [[0.0] * r for _ in range(n)]
    for i in range(n):
        for j in range(r):
            acc = 0.0
            for k in range(d_model):
                acc += x[i][k] * W_down_kv[k][j]
            c_kv[i][j] = acc

    Q = [[0.0] * dim_q for _ in range(n)]
    for i in range(n):
        for j in range(dim_q):
            acc = 0.0
            for k in range(d_model):
                acc += x[i][k] * W_Q[k][j]
            Q[i][j] = acc

    K = [[0.0] * dim_kv for _ in range(n)]
    for i in range(n):
        for j in range(dim_kv):
            acc = 0.0
            for k in range(r):
                acc += c_kv[i][k] * W_up_K[k][j]
            K[i][j] = acc

    V = [[0.0] * dim_kv for _ in range(n)]
    for i in range(n):
        for j in range(dim_kv):
            acc = 0.0
            for k in range(r):
                acc += c_kv[i][k] * W_up_V[k][j]
            V[i][j] = acc

    Qh = _split_heads(Q, num_heads)
    Kh = _split_heads(K, num_heads)
    Vh = _split_heads(V, num_heads)
    d_head = len(Qh[0][0])
    scale = 1.0 / math.sqrt(d_head)

    scores = [[[0.0] * n for _ in range(n)] for _ in range(num_heads)]
    for h in range(num_heads):
        for i in range(n):
            for j in range(n):
                acc = 0.0
                for k in range(d_head):
                    acc += Qh[h][i][k] * Kh[h][j][k]
                scores[h][i][j] = acc * scale

    for h in range(num_heads):
        for i in range(n):
            max_val = scores[h][i][0]
            for j in range(1, n):
                if scores[h][i][j] > max_val:
                    max_val = scores[h][i][j]
            for j in range(n):
                scores[h][i][j] = scores[h][i][j] - max_val

    w = [[[0.0] * n for _ in range(n)] for _ in range(num_heads)]
    for h in range(num_heads):
        for i in range(n):
            sum_exp = 0.0
            for j in range(n):
                val = math.exp(scores[h][i][j])
                w[h][i][j] = val
                sum_exp += val
            for j in range(n):
                w[h][i][j] = w[h][i][j] / sum_exp

    out_h = [[[0.0] * d_head for _ in range(n)] for _ in range(num_heads)]
    for h in range(num_heads):
        for i in range(n):
            for j in range(d_head):
                acc = 0.0
                for k in range(n):
                    acc += w[h][i][k] * Vh[h][k][j]
                out_h[h][i][j] = acc

    out = [[0.0] * (num_heads * d_head) for _ in range(n)]
    for h in range(num_heads):
        for i in range(n):
            for k in range(d_head):
                out[i][h * d_head + k] = out_h[h][i][k]

    return out, c_kv
