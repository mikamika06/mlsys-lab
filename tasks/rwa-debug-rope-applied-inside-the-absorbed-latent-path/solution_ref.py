def _rope(x, cos, sin):
    n = len(x)
    h = len(x[0])
    half = h // 2
    out = []
    for i in range(n):
        row = []
        x_row = x[i]
        c_row = cos[i]
        s_row = sin[i]
        for j in range(half):
            x0 = x_row[2 * j]
            x1 = x_row[2 * j + 1]
            c = c_row[j]
            s = s_row[j]
            row.append(x0 * c - x1 * s)
            row.append(x0 * s + x1 * c)
        out.append(row)
    return out


def _matmul(A, B):
    n = len(A)
    k = len(A[0])
    m = len(B[0])
    C = [[0.0 for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for kj in range(k):
            aik = A[i][kj]
            if aik == 0.0:
                continue
            b_row = B[kj]
            c_row = C[i]
            for j in range(m):
                c_row[j] += aik * b_row[j]
    return C


def mla_kv_features(z, head, w_latent, w_head, cos, sin):
    latent = _matmul(z, w_latent)
    decoupled = _matmul(head, w_head)
    decoupled = _rope(decoupled, cos, sin)
    return [l + d for l, d in zip(latent, decoupled)]
