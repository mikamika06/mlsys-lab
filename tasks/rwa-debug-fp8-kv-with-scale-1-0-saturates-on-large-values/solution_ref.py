import math
import numpy as np


def _e4m3_roundtrip(x, scale):
    x_arr = np.asarray(x, dtype=np.float64)
    out = np.zeros(x_arr.shape, dtype=np.float64)
    for i in range(x_arr.size):
        y = float(x_arr.flat[i]) / scale
        if y > 448.0:
            y = 448.0
        elif y < -448.0:
            y = -448.0
        if y > 0.0:
            sign = 1.0
        elif y < 0.0:
            sign = -1.0
        else:
            sign = 0.0
        ay = abs(y)
        exp_val = math.floor(math.log2(max(ay, 2.0 ** -9)))
        if exp_val > 7:
            exp_val = 7
        elif exp_val < -6:
            exp_val = -6
        two_exp = 2.0 ** exp_val
        frac = ay / two_exp - 1.0
        mant = round(frac * 8.0) / 8.0
        val = (1.0 + mant) * two_exp
        if ay < 2.0 ** -6:
            val = round(ay / (2.0 ** -9)) * (2.0 ** -9)
        if ay == 0.0:
            val = 0.0
        out.flat[i] = sign * val * scale
    return out


def fp8_attention_output(Q, K, V):
    max_k = 0.0
    for i in range(K.size):
        a = abs(float(K.flat[i]))
        if a > max_k:
            max_k = a
    sk = max_k / 448.0
    if sk < 1e-12:
        sk = 1e-12

    max_v = 0.0
    for i in range(V.size):
        a = abs(float(V.flat[i]))
        if a > max_v:
            max_v = a
    sv = max_v / 448.0
    if sv < 1e-12:
        sv = 1e-12

    K_hat = _e4m3_roundtrip(K, sk)
    V_hat = _e4m3_roundtrip(V, sv)

    N = Q.shape[0]
    d = Q.shape[1]
    M = K_hat.shape[0]
    d_v = V_hat.shape[1]

    sqrt_d = math.sqrt(d)

    scores = np.zeros((N, M), dtype=np.float64)
    for i in range(N):
        for j in range(M):
            acc = 0.0
            for k in range(d):
                acc += float(Q[i, k]) * float(K_hat[j, k])
            scores[i, j] = acc / sqrt_d

    for i in range(N):
        row_max = float("-inf")
        for j in range(M):
            if scores[i, j] > row_max:
                row_max = scores[i, j]
        row_sum = 0.0
        for j in range(M):
            val = math.exp(scores[i, j] - row_max)
            scores[i, j] = val
            row_sum += val
        for j in range(M):
            scores[i, j] /= row_sum

    probs = scores

    out = np.zeros((N, d_v), dtype=np.float64)
    for i in range(N):
        for k in range(d_v):
            acc = 0.0
            for j in range(M):
                acc += probs[i, j] * float(V_hat[j, k])
            out[i, k] = acc

    return out.astype(np.float64)
