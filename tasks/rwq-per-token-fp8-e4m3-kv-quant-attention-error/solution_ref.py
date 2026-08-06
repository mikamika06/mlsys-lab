import math
import numpy as np


def _fp8_e4m3_round(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    shape = x.shape
    for idx in np.ndindex(shape):
        val = x[idx]
        sign = -1.0 if val < 0.0 else (1.0 if val > 0.0 else 0.0)
        ax = abs(val)
        if ax >= 2.0 ** -6:
            exp = math.floor(math.log2(ax))
            if exp < -6:
                exp = -6
            elif exp > 8:
                exp = 8
            step = 2.0 ** (exp - 3)
            mant = round(ax / step) * step
            if mant > 448.0:
                mant = 448.0
            out[idx] = sign * mant
        else:
            mant = round(ax / (2.0 ** -9)) * (2.0 ** -9)
            out[idx] = sign * mant
    return out


def _quantize_per_token(x):
    x = np.asarray(x, dtype=np.float64)
    rows, cols = x.shape
    scale = np.zeros((rows, 1), dtype=np.float64)
    for i in range(rows):
        max_val = 0.0
        for j in range(cols):
            val = abs(x[i, j])
            if val > max_val:
                max_val = val
        s = max_val / 448.0
        if s == 0.0:
            scale[i, 0] = 1.0
        else:
            scale[i, 0] = s

    divided = np.zeros_like(x)
    for i in range(rows):
        s = scale[i, 0]
        for j in range(cols):
            divided[i, j] = x[i, j] / s

    rounded = _fp8_e4m3_round(divided)
    out = np.zeros_like(x)
    for i in range(rows):
        s = scale[i, 0]
        for j in range(cols):
            out[i, j] = rounded[i, j] * s
    return out


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    rows, cols = x.shape
    max_vals = np.zeros((rows, 1), dtype=np.float64)
    for i in range(rows):
        m = x[i, 0]
        for j in range(1, cols):
            if x[i, j] > m:
                m = x[i, j]
        max_vals[i, 0] = m

    exp_x = np.zeros_like(x)
    for i in range(rows):
        m = max_vals[i, 0]
        for j in range(cols):
            exp_x[i, j] = math.exp(x[i, j] - m)

    sum_exp = np.zeros((rows, 1), dtype=np.float64)
    for i in range(rows):
        s = 0.0
        for j in range(cols):
            s += exp_x[i, j]
        sum_exp[i, 0] = s

    out = np.zeros_like(x)
    for i in range(rows):
        s = sum_exp[i, 0]
        for j in range(cols):
            out[i, j] = exp_x[i, j] / s
    return out


def fp8_kv_attention(Q, K, V):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    K_hat = _quantize_per_token(K)
    V_hat = _quantize_per_token(V)

    d_k = Q.shape[1]
    sqrt_dk = math.sqrt(d_k)

    def matmul(A, B):
        r1, c1 = A.shape
        r2, c2 = B.shape
        res = np.zeros((r1, c2), dtype=np.float64)
        for i in range(r1):
            for k in range(c1):
                aik = A[i, k]
                if aik == 0.0:
                    continue
                for j in range(c2):
                    res[i, j] += aik * B[k, j]
        return res

    KT = K.T
    K_hat_T = K_hat.T

    qk1 = matmul(Q, KT)
    for i in range(qk1.shape[0]):
        for j in range(qk1.shape[1]):
            qk1[i, j] /= sqrt_dk
    full = matmul(_softmax(qk1), V)

    qk2 = matmul(Q, K_hat_T)
    for i in range(qk2.shape[0]):
        for j in range(qk2.shape[1]):
            qk2[i, j] /= sqrt_dk
    out = matmul(_softmax(qk2), V_hat)

    diff = out - full
    diff_sq_sum = 0.0
    total_elements = diff.size
    for idx in np.ndindex(diff.shape):
        val = diff[idx]
        diff_sq_sum += val * val
    mse = float(diff_sq_sum / total_elements)

    return out.astype(np.float64), mse
