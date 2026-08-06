import math
import numpy as np

def rope_relative_dot(q, k, pos_q, pos_k):
    d = q.shape[0]
    if d % 2 != 0:
        raise ValueError("q and k must have even length")
    q_rot = np.empty_like(q)
    k_rot = np.empty_like(k)
    for i in range(0, d, 2):
        inv_freq = 1.0 / (10000.0 ** (i / d))
        pos_q_val = pos_q * inv_freq
        pos_k_val = pos_k * inv_freq
        sin_q = math.sin(pos_q_val)
        cos_q = math.cos(pos_q_val)
        sin_k = math.sin(pos_k_val)
        cos_k = math.cos(pos_k_val)
        q_rot[i] = q[i] * cos_q - q[i + 1] * sin_q
        q_rot[i + 1] = q[i] * sin_q + q[i + 1] * cos_q
        k_rot[i] = k[i] * cos_k - k[i + 1] * sin_k
        k_rot[i + 1] = k[i] * sin_k + k[i + 1] * cos_k

    acc = 0.0
    for j in range(d):
        acc += q_rot[j] * k_rot[j]
    return float(acc)
