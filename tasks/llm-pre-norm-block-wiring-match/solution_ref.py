import math
import numpy as np


def _matmul(A, B):
    M = A.shape[0]
    K = A.shape[1]
    N = B.shape[1]
    out = np.empty((M, N), dtype=np.float64)
    for i in range(M):
        for j in range(N):
            s = 0.0
            for k in range(K):
                s += A[i, k] * B[k, j]
            out[i, j] = s
    return out


def _layer_norm(z, gamma, beta, eps=1e-5):
    T = z.shape[0]
    d = z.shape[1]
    out = np.empty((T, d), dtype=np.float64)
    for i in range(T):
        s = 0.0
        for j in range(d):
            s += z[i, j]
        mu = s / d
        sq_err = 0.0
        for j in range(d):
            diff = z[i, j] - mu
            sq_err += diff * diff
        var = sq_err / d
        inv_std = math.sqrt(var + eps)
        for j in range(d):
            out[i, j] = gamma[j] * ((z[i, j] - mu) / inv_std) + beta[j]
    return out


def _softmax(s):
    rows = s.shape[0]
    cols = s.shape[1]
    out = np.empty((rows, cols), dtype=np.float64)
    for i in range(rows):
        max_val = s[i, 0]
        for j in range(1, cols):
            if s[i, j] > max_val:
                max_val = s[i, j]
        e_row = [math.exp(s[i, j] - max_val) for j in range(cols)]
        sum_exp = 0.0
        for j in range(cols):
            sum_exp += e_row[j]
        for j in range(cols):
            out[i, j] = e_row[j] / sum_exp
    return out


def _attn(z, Wq, Wk, Wv, Wo):
    d = z.shape[-1]
    Q = _matmul(z, Wq)
    K = _matmul(z, Wk)
    V = _matmul(z, Wv)
    T = z.shape[0]
    scores = np.empty((T, T), dtype=np.float64)
    scale = math.sqrt(d)
    for i in range(T):
        for j in range(T):
            s = 0.0
            for k in range(d):
                s += Q[i, k] * K[j, k]
            scores[i, j] = s / scale
    return _matmul(_matmul(_softmax(scores), V), Wo)


def _gelu(u):
    rows = u.shape[0]
    cols = u.shape[1]
    out = np.empty((rows, cols), dtype=np.float64)
    coeff = math.sqrt(2.0 / math.pi)
    for i in range(rows):
        for j in range(cols):
            val = u[i, j]
            inner = coeff * (val + 0.044715 * (val ** 3))
            out[i, j] = 0.5 * val * (1.0 + math.tanh(inner))
    return out


def _mlp(z, W1, b1, W2, b2):
    T = z.shape[0]
    d = z.shape[1]
    h = W1.shape[1]
    hidden = np.empty((T, h), dtype=np.float64)
    for i in range(T):
        for j in range(h):
            s = 0.0
            for k in range(d):
                s += z[i, k] * W1[k, j]
            hidden[i, j] = s + b1[j]
    act = _gelu(hidden)
    out = np.empty((T, d), dtype=np.float64)
    for i in range(T):
        for j in range(d):
            s = 0.0
            for k in range(h):
                s += act[i, k] * W2[k, j]
            out[i, j] = s + b2[j]
    return out


def pre_norm_block(x, gamma1, beta1, gamma2, beta2,
                   Wq, Wk, Wv, Wo, W1, b1, W2, b2):
    """One pre-norm transformer block over a residual stream x of shape (T, d)."""
    x = np.asarray(x, dtype=np.float64)
    ln1 = _layer_norm(x, gamma1, beta1)
    attn_out = _attn(ln1, Wq, Wk, Wv, Wo)
    T = x.shape[0]
    d = x.shape[1]
    h = np.empty((T, d), dtype=np.float64)
    for i in range(T):
        for j in range(d):
            h[i, j] = x[i, j] + attn_out[i, j]
    ln2 = _layer_norm(h, gamma2, beta2)
    mlp_out = _mlp(ln2, W1, b1, W2, b2)
    y = np.empty((T, d), dtype=np.float64)
    for i in range(T):
        for j in range(d):
            y[i, j] = h[i, j] + mlp_out[i, j]
    return y
