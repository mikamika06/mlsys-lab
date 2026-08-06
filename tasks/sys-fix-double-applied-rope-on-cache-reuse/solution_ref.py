import math
import numpy as np


def rope(x, positions, base: float = 10000.0):
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    positions = np.atleast_1d(np.asarray(positions, dtype=np.float64))
    B, d = x.shape
    d_half = d // 2
    theta = [base ** (-2.0 * i / d) for i in range(d_half)]
    out_list = []
    for b in range(B):
        pos_val = positions[b]
        row = [0.0] * d
        for j in range(d_half):
            angle = pos_val * theta[j]
            c = math.cos(angle)
            s = math.sin(angle)
            x_even = x[b, 2 * j]
            x_odd = x[b, 2 * j + 1]
            row[2 * j] = x_even * c - x_odd * s
            row[2 * j + 1] = x_even * s + x_odd * c
        out_list.append(row)
    return np.array(out_list, dtype=np.float64)


def _softmax(z):
    z_list = [val for val in z]
    max_z = z_list[0]
    for val in z_list[1:]:
        if val > max_z:
            max_z = val
    e_list = [math.exp(val - max_z) for val in z_list]
    sum_e = 0.0
    for val in e_list:
        sum_e += val
    return np.array([val / sum_e for val in e_list], dtype=np.float64)


def decode_step(cache, k_raw, v, q_raw, pos):
    d = np.asarray(k_raw).shape[-1]
    k = rope(k_raw, pos)[0]
    q = rope(q_raw, pos)[0]
    k_list = cache["k"].tolist() if hasattr(cache["k"], "tolist") else list(cache["k"])
    k_list.append(k.tolist() if hasattr(k, "tolist") else list(k))
    cache["k"] = np.array(k_list, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    v_list = cache["v"].tolist() if hasattr(cache["v"], "tolist") else list(cache["v"])
    v_list.append(v_arr.tolist() if hasattr(v_arr, "tolist") else list(v_arr))
    cache["v"] = np.array(v_list, dtype=np.float64)
    num_tokens = cache["k"].shape[0]
    scores_list = []
    sqrt_d = math.sqrt(d)
    for t in range(num_tokens):
        dot_val = 0.0
        for j in range(d):
            dot_val += cache["k"][t, j] * q[j]
        scores_list.append(dot_val / sqrt_d)
    w = _softmax(scores_list)
    out_list = [0.0] * d
    for j in range(d):
        s_val = 0.0
        for t in range(num_tokens):
            s_val += w[t] * cache["v"][t, j]
        out_list[j] = s_val
    out = np.array(out_list, dtype=np.float64)
    return out, cache
