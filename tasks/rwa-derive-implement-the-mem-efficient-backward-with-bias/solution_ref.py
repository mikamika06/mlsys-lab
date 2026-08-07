import math


def biased_flash_backward(
    Q: list[list[float]],
    K: list[list[float]],
    V: list[list[float]],
    B: list[list[float]],
    dO: list[list[float]],
    m: list[float],
    l: list[float],
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    """Memory-efficient attention backward with an additive bias (e.g.
    ALiBi or a mask bias), given only the saved row statistics (m, l)
    from the forward pass -- never a cached probability matrix.

    Q, K, V   : (n, d)
    B         : (n, n) additive bias, added to the scaled scores before
                softmax on the forward pass. Fixed (no gradient wanted).
    dO        : (n, d) upstream gradient w.r.t. the forward output O.
    m, l      : (n,) row max and row softmax-normalizer saved during the
                forward pass, i.e. S = Q@K.T/sqrt(d) + B,
                m = rowmax(S), l = rowsum(exp(S - m)).

    Returns (dQ, dK, dV), each shaped like Q, K, V respectively.
    """
    n = len(Q)
    d = len(Q[0])
    scale = math.sqrt(float(d))

    P = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s_val = 0.0
            for k in range(d):
                s_val += Q[i][k] * K[j][k]
            s_val = s_val / scale + B[i][j]
            P[i][j] = math.exp(s_val - m[i]) / l[i]

    dV = [[0.0] * d for _ in range(n)]
    for i in range(n):
        for k in range(d):
            val = 0.0
            for j in range(n):
                val += P[j][i] * dO[j][k]
            dV[i][k] = val

    dP = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            val = 0.0
            for k in range(d):
                val += dO[i][k] * V[j][k]
            dP[i][j] = val

    dS = [[0.0] * n for _ in range(n)]
    for i in range(n):
        corr = 0.0
        for j in range(n):
            corr += dP[i][j] * P[i][j]
        for j in range(n):
            dS[i][j] = P[i][j] * (dP[i][j] - corr)

    dQ = [[0.0] * d for _ in range(n)]
    for i in range(n):
        for k in range(d):
            val = 0.0
            for j in range(n):
                val += dS[i][j] * K[j][k]
            dQ[i][k] = val / scale

    dK = [[0.0] * d for _ in range(n)]
    for j in range(n):
        for k in range(d):
            val = 0.0
            for i in range(n):
                val += dS[i][j] * Q[i][k]
            dK[j][k] = val / scale

    return (dQ, dK, dV)
