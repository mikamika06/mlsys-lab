import math
import numpy as np


def _layer_norm(x, gamma, beta, eps=1e-5):
    x_arr = np.asarray(x, dtype=np.float64)
    shape = x_arr.shape
    d = shape[-1]
    flat_x = x_arr.reshape(-1, d)
    num_rows = flat_x.shape[0]
    out_flat = np.zeros((num_rows, d), dtype=np.float64)
    for i in range(num_rows):
        s = 0.0
        for j in range(d):
            s += flat_x[i, j]
        mean = s / d
        v_sum = 0.0
        for j in range(d):
            diff = flat_x[i, j] - mean
            v_sum += diff * diff
        var = v_sum / d
        denom = math.sqrt(var + eps)
        for j in range(d):
            normed = (flat_x[i, j] - mean) / denom
            out_flat[i, j] = normed * gamma[j] + beta[j]
    return np.asarray(out_flat.reshape(shape), dtype=np.float64)


def transformer_block(x, w_attn, w_ff, gamma, beta):
    x_arr = np.asarray(x, dtype=np.float64)
    w_attn_arr = np.asarray(w_attn, dtype=np.float64)
    w_ff_arr = np.asarray(w_ff, dtype=np.float64)
    gamma_arr = np.asarray(gamma, dtype=np.float64)
    beta_arr = np.asarray(beta, dtype=np.float64)

    ln1 = _layer_norm(x_arr, gamma_arr, beta_arr)
    
    shape_ln1 = ln1.shape
    flat_ln1 = ln1.reshape(-1, shape_ln1[-1])
    r1, c1 = flat_ln1.shape
    r2, c2 = w_attn_arr.shape
    
    attn_out_flat = np.zeros((r1, c2), dtype=np.float64)
    for i in range(r1):
        for j in range(c2):
            s = 0.0
            for k in range(c1):
                s += flat_ln1[i, k] * w_attn_arr[k, j]
            attn_out_flat[i, j] = s
    attn_out = attn_out_flat.reshape(shape_ln1[:-1] + (c2,))

    h1 = x_arr + attn_out

    ln2 = _layer_norm(h1, gamma_arr, beta_arr)
    shape_ln2 = ln2.shape
    flat_ln2 = ln2.reshape(-1, shape_ln2[-1])
    r3, c3 = flat_ln2.shape
    r4, c4 = w_ff_arr.shape

    ff_out_flat = np.zeros((r3, c4), dtype=np.float64)
    for i in range(r3):
        for j in range(c4):
            s = 0.0
            for k in range(c3):
                s += flat_ln2[i, k] * w_ff_arr[k, j]
            ff_out_flat[i, j] = s
    ff_out = ff_out_flat.reshape(shape_ln2[:-1] + (c4,))

    y = h1 + ff_out
    return np.asarray(y, dtype=np.float64)
