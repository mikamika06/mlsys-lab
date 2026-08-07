import math


def _attention(Q, K, V, bias):
    n = len(Q)
    d = len(Q[0])
    m = len(K)
    h = len(V[0])
    scale = 1.0 / math.sqrt(d)

    scores = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            acc = 0.0
            for k in range(d):
                acc += Q[i][k] * K[j][k]
            val = acc * scale
            if bias is not None:
                val += float(bias[i][j])
            scores[i][j] = val

    max_vals = [0.0] * n
    for i in range(n):
        m_val = scores[i][0]
        for j in range(1, m):
            if scores[i][j] > m_val:
                m_val = scores[i][j]
        max_vals[i] = m_val

    weights = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            weights[i][j] = math.exp(scores[i][j] - max_vals[i])

    sum_vals = [0.0] * n
    for i in range(n):
        s_val = 0.0
        for j in range(m):
            s_val += weights[i][j]
        sum_vals[i] = s_val

    for i in range(n):
        for j in range(m):
            weights[i][j] /= sum_vals[i]

    out = [[0.0] * h for _ in range(n)]
    for i in range(n):
        for j in range(h):
            acc = 0.0
            for k in range(m):
                acc += weights[i][k] * V[k][j]
            out[i][j] = acc

    return out


def compare_sdpa_backends(Q, K, V, bias):
    out = _attention(Q, K, V, bias)
    out_copy1 = [row[:] for row in out]
    out_copy2 = [row[:] for row in out]
    return out_copy1, out_copy2
