import math
import numpy as np


def _dot_acc(a, b, dtype):
    acc = dtype(0.0)
    for x, y in zip(a, b):
        acc = dtype(acc + dtype(dtype(x) * dtype(y)))
    return acc


def _attention_precision(Q, K, V, dtype):
    Q = np.asarray(Q, dtype=np.float64).astype(dtype)
    K = np.asarray(K, dtype=np.float64).astype(dtype)
    V = np.asarray(V, dtype=np.float64).astype(dtype)

    n, d = Q.shape
    m, dv = V.shape
    scale = dtype(1.0 / math.sqrt(d))

    O = np.zeros((n, dv), dtype=np.float64)
    for i in range(n):
        S = np.empty(m, dtype=dtype)
        for j in range(m):
            s = _dot_acc(Q[i], K[j], dtype)
            S[j] = dtype(s * scale)

        m_i = dtype(S[0])
        for j in range(1, m):
            if S[j] > m_i:
                m_i = dtype(S[j])

        exp_vals = np.empty(m, dtype=dtype)
        for j in range(m):
            shifted = dtype(S[j] - m_i)
            exp_vals[j] = dtype(math.exp(float(shifted)))

        l_i = dtype(0.0)
        for j in range(m):
            l_i = dtype(l_i + exp_vals[j])

        P = np.empty(m, dtype=dtype)
        for j in range(m):
            P[j] = dtype(exp_vals[j] / l_i)

        for k in range(dv):
            col = V[:, k]
            o = dtype(0.0)
            for j in range(m):
                o = dtype(o + dtype(P[j] * col[j]))
            O[i, k] = float(o)

    return O


def _rel_err(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    diff_sq_acc = 0.0
    a_sq_acc = 0.0
    for x, y in zip(a, b):
        diff = float(y) - float(x)
        diff_sq_acc += diff * diff
        a_sq_acc += float(x) * float(x)
    return float(math.sqrt(diff_sq_acc) / (math.sqrt(a_sq_acc) + 1e-12))


def fp16_vs_fp32_attention_error(Q, K, V):
    O_ref = _attention_precision(Q, K, V, np.float64)
    O_16 = _attention_precision(Q, K, V, np.float16)
    O_32 = _attention_precision(Q, K, V, np.float32)
    return _rel_err(O_ref, O_16), _rel_err(O_ref, O_32)
