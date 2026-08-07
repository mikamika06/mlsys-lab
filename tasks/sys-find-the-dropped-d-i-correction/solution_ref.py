import math


def flash_attention_backward(Q: list[list[float]], K: list[list[float]], V: list[list[float]],
                              O: list[list[float]], L: list[float], dO: list[list[float]],
                              scale: float):
    """
    Recompute-based FlashAttention backward: recompute P from Q, K, L
    (never read a stored full attention matrix), then apply the softmax
    VJP with the D_i = rowsum(dO * O) correction term using pure Python.
    """
    n = len(Q)
    d = len(Q[0])
    k_len = len(K)

    # Matrix multiplication: S = (Q @ K.T) * scale
    S = []
    for i in range(n):
        row = []
        for j in range(k_len):
            dot = sum(Q[i][c] * K[j][c] for c in range(d))
            row.append(dot * scale)
        S.append(row)

    # Recomputed attention weights: P_ij = exp(S_ij - L_i)
    P = []
    for i in range(n):
        row = []
        for j in range(k_len):
            row.append(math.exp(S[i][j] - L[i]))
        P.append(row)

    # dV = P.T @ dO
    d_v_cols = len(dO[0])
    dV = [[0.0.real for _ in range(d_v_cols)] for _ in range(k_len)]
    for j in range(k_len):
        for c in range(d_v_cols):
            val = 0.0
            for i in range(n):
                val += P[i][j] * dO[i][c]
            dV[j][c] = val

    # dP = dO @ V.T
    v_rows = len(V)
    dP = [[0.0 for _ in range(v_rows)] for _ in range(n)]
    for i in range(n):
        for j in range(v_rows):
            val = 0.0
            for c in range(d):
                val += dO[i][c] * V[j][c]
            dP[i][j] = val

    # D = rowsum(dO * O)
    D = []
    for i in range(n):
        val = sum(dO[i][c] * O[i][c] for c in range(d))
        D.append(val)

    # dS = P * (dP - D[:, None])
    dS = []
    for i in range(n):
        row = []
        for j in range(k_len):
            row.append(P[i][j] * (dP[i][j] - D[i]))
        dS.append(row)

    # dQ = (dS @ K) * scale
    dQ = [[0.0 for _ in range(d)] for _ in range(n)]
    for i in range(n):
        for c in range(d):
            val = 0.0
            for j in range(k_len):
                val += dS[i][j] * K[j][c]
            dQ[i][c] = val * scale

    # dK = (dS.T @ Q) * scale
    d_k_cols = len(Q[0])
    dK = [[0.0 for _ in range(d_k_cols)] for _ in range(k_len)]
    for j in range(k_len):
        for c in range(d_k_cols):
            val = 0.0
            for i in range(n):
                val += dS[i][j] * Q[i][c]
            dK[j][c] = val * scale

    return dQ, dK, dV
