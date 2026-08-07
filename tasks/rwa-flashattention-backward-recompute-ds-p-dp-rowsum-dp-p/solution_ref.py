import math


def flash_attention_backward(
    q: list[list[float]],
    k: list[list[float]],
    v: list[list[float]],
    do: list[list[float]],
    m: list[list[float]],
    l: list[list[float]],
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    N_q = len(q)
    D = len(q[0]) if N_q > 0 else 0
    N_k = len(k)

    scores = [[0.0] * N_k for _ in range(N_q)]
    for i in range(N_q):
        for j in range(N_k):
            acc = 0.0
            for d in range(D):
                acc += q[i][d] * k[j][d]
            scores[i][j] = acc

    p = [[0.0] * N_k for _ in range(N_q)]
    for i in range(N_q):
        for j in range(N_k):
            p[i][j] = math.exp(scores[i][j] - m[i][0]) / l[i][0]

    dp = [[0.0] * N_k for _ in range(N_q)]
    for i in range(N_q):
        for j in range(N_k):
            acc = 0.0
            for d in range(D):
                acc += do[i][d] * v[j][d]
            dp[i][j] = acc

    rowsum = [[0.0] for _ in range(N_q)]
    for i in range(N_q):
        acc = 0.0
        for j in range(N_k):
            acc += dp[i][j] * p[i][j]
        rowsum[i][0] = acc

    ds = [[0.0] * N_k for _ in range(N_q)]
    for i in range(N_q):
        for j in range(N_k):
            ds[i][j] = p[i][j] * (dp[i][j] - rowsum[i][0])

    dq = [[0.0] * D for _ in range(N_q)]
    for i in range(N_q):
        for d in range(D):
            acc = 0.0
            for j in range(N_k):
                acc += ds[i][j] * k[j][d]
            dq[i][d] = acc

    dk = [[0.0] * D for _ in range(N_k)]
    for j in range(N_k):
        for d in range(D):
            acc = 0.0
            for i in range(N_q):
                acc += ds[i][j] * q[i][d]
            dk[j][d] = acc

    dv = [[0.0] * D for _ in range(N_k)]
    for j in range(N_k):
        for d in range(D):
            acc = 0.0
            for i in range(N_q):
                acc += p[i][j] * do[i][d]
            dv[j][d] = acc

    return dq, dk, dv
