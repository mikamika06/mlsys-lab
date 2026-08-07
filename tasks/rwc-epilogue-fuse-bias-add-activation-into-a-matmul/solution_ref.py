def fused_matmul_epilogue(A, B, bias, activation, out):
    m, k = len(A), len(A[0])
    n = len(B[0])

    for i in range(m):
        for j in range(n):
            acc = 0.0
            for r in range(k):
                acc += A[i][r] * B[r][j]
            value = acc + bias[j]
            if activation == "relu":
                value = max(0.0, value)
            elif activation != "identity":
                raise ValueError("unknown activation")
            out[i][j] = value

    return out
