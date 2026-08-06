import math


def _matmul(A, B):
    M = len(A)
    K = len(A[0])
    N = len(B[0])
    out = [[0.0] * N for _ in range(M)]
    for i in range(M):
        for j in range(N):
            s = 0.0
            for k in range(K):
                s += A[i][k] * B[k][j]
            out[i][j] = s
    return out


def _layer_norm(z, gamma, beta, eps=1e-5):
    T = len(z)
    d = len(z[0])
    out = [[0.0] * d for _ in range(T)]
    for i in range(T):
        s = 0.0
        for j in range(d):
            s += z[i][j]
        mu = s / d
        sq_err = 0.0
        for j in range(d):
            diff = z[i][j] - mu
            sq_err += diff * diff
        var = sq_err / d
        inv_std = math.sqrt(var + eps)
        for j in range(d):
            out[i][j] = gamma[j] * ((z[i][j] - mu) / inv_std) + beta[j]
    return out


def _softmax(s):
    rows = len(s)
    cols = len(s[0])
    out = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        max_val = s[i][0]
        for j in range(1, cols):
            if s[i][j] > max_val:
                max_val = s[i][j]
        e_row = [math.exp(s[i][j] - max_val) for j in range(cols)]
        sum_exp = 0.0
        for j in range(cols):
            sum_exp += e_row[j]
        for j in range(cols):
            out[i][j] = e_row[j] / sum_exp
    return out


def _attn(z, Wq, Wk, Wv, Wo):
    d = len(z[0])
    Q = _matmul(z, Wq)
    K = _matmul(z, Wk)
    V = _matmul(z, Wv)
    T = len(z)
    scores = [[0.0] * T for _ in range(T)]
    scale = math.sqrt(d)
    for i in range(T):
        for j in range(T):
            s = 0.0
            for k in range(d):
                s += Q[i][k] * K[j][k]
            scores[i][j] = s / scale
    return _matmul(_matmul(_softmax(scores), V), Wo)


def _gelu(u):
    rows = len(u)
    cols = len(u[0])
    out = [[0.0] * cols for _ in range(rows)]
    coeff = math.sqrt(2.0 / math.pi)
    for i in range(rows):
        for j in range(cols):
            val = u[i][j]
            inner = coeff * (val + 0.044715 * (val ** 3))
            out[i][j] = 0.5 * val * (1.0 + math.tanh(inner))
    return out


def _mlp(z, W1, b1, W2, b2):
    T = len(z)
    d = len(z[0])
    h = len(W1[0])
    hidden = [[0.0] * h for _ in range(T)]
    for i in range(T):
        for j in range(h):
            s = 0.0
            for k in range(d):
                s += z[i][k] * W1[k][j]
            hidden[i][j] = s + b1[j]
    act = _gelu(hidden)
    out = [[0.0] * d for _ in range(T)]
    for i in range(T):
        for j in range(d):
            s = 0.0
            for k in range(h):
                s += act[i][k] * W2[k][j]
            out[i][j] = s + b2[j]
    return out


def pre_norm_block(
    x: list[list[float]],
    gamma1: list[float],
    beta1: list[float],
    gamma2: list[float],
    beta2: list[float],
    Wq: list[list[float]],
    Wk: list[list[float]],
    Wv: list[list[float]],
    Wo: list[list[float]],
    W1: list[list[float]],
    b1: list[float],
    W2: list[list[float]],
    b2: list[float],
) -> list[list[float]]:
    """One pre-norm transformer block over a residual stream x of shape (T, d)."""
    ln1 = _layer_norm(x, gamma1, beta1)
    attn_out = _attn(ln1, Wq, Wk, Wv, Wo)
    T = len(x)
    d = len(x[0])
    h = [[0.0] * d for _ in range(T)]
    for i in range(T):
        for j in range(d):
            h[i][j] = x[i][j] + attn_out[i][j]
    ln2 = _layer_norm(h, gamma2, beta2)
    mlp_out = _mlp(ln2, W1, b1, W2, b2)
    y = [[0.0] * d for _ in range(T)]
    for i in range(T):
        for j in range(d):
            y[i][j] = h[i][j] + mlp_out[i][j]
    return y
