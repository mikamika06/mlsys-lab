def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    n = len(A)
    d = len(A[0])
    m = len(B[0])
    out = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for k in range(d):
            a_ik = A[i][k]
            b_k = B[k]
            out_i = out[i]
            for j in range(m):
                out_i[j] += a_ik * b_k[j]
    return out


def fused_qkv_projection(
    X: list[list[float]],
    Wq: list[list[float]],
    Wk: list[list[float]],
    Wv: list[list[float]],
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    d = len(Wq)
    m = len(Wq[0])

    W_qkv = []
    for i in range(d):
        W_qkv.append(Wq[i] + Wk[i] + Wv[i])

    QKV = matmul(X, W_qkv)

    n = len(X)
    Q = [[0.0] * m for _ in range(n)]
    K = [[0.0] * m for _ in range(n)]
    V = [[0.0] * m for _ in range(n)]

    for i in range(n):
        row = QKV[i]
        Q[i] = row[:m]
        K[i] = row[m:2 * m]
        V[i] = row[2 * m:3 * m]

    return Q, K, V
