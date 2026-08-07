import math


def gqa_attention(Q: list[list[list[float]]], K: list[list[list[float]]], V: list[list[list[float]]]) -> list[list[list[float]]]:
    """
    Grouped-query attention: query heads are split into n_kv contiguous
    groups of size g = n_q // n_kv, each group sharing one KV head.
    """
    n_q = len(Q)
    n = len(Q[0])
    d = len(Q[0][0])
    n_kv = len(K)
    g = n_q // n_kv
    scale = 1.0 / math.sqrt(d)

    O = [[[0.0 for _ in range(d)] for _ in range(n)] for _ in range(n_q)]

    for h in range(n_q):
        kv = h // g
        Q_h = Q[h]
        K_kv = K[kv]
        V_kv = V[kv]

        S = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                acc = 0.0
                for k in range(d):
                    acc += Q_h[i][k] * K_kv[j][k]
                S[i][j] = acc * scale

        max_S = [0.0 for _ in range(n)]
        for i in range(n):
            m = S[i][0]
            for j in range(1, n):
                if S[i][j] > m:
                    m = S[i][j]
            max_S[i] = m

        P = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                P[i][j] = math.exp(S[i][j] - max_S[i])

        sum_P = [0.0 for _ in range(n)]
        for i in range(n):
            s_acc = 0.0
            for j in range(n):
                s_acc += P[i][j]
            sum_P[i] = s_acc

        for i in range(n):
            for j in range(n):
                P[i][j] /= sum_P[i]

        O_h = [[0.0 for _ in range(d)] for _ in range(n)]
        for i in range(n):
            for k in range(d):
                acc = 0.0
                for j in range(n):
                    acc += P[i][j] * V_kv[j][k]
                O_h[i][k] = acc

        for i in range(n):
            for k in range(d):
                O[h][i][k] = O_h[i][k]

    return O
