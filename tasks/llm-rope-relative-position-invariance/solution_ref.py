import numpy as np

def rope_relative_dot(q, k, pos_q, pos_k):
    d = q.shape[0]
    if d % 2 != 0:
        raise ValueError("q and k must have even length")
    inv_freq = 1 / (10000 ** (np.arange(0, d, 2) / d))
    pos_q_vec = pos_q * inv_freq
    pos_k_vec = pos_k * inv_freq
    sin_q = np.sin(pos_q_vec)
    cos_q = np.cos(pos_q_vec)
    sin_k = np.sin(pos_k_vec)
    cos_k = np.cos(pos_k_vec)

    q_rot = np.empty_like(q)
    q_rot[0::2] = q[0::2] * cos_q - q[1::2] * sin_q
    q_rot[1::2] = q[0::2] * sin_q + q[1::2] * cos_q

    k_rot = np.empty_like(k)
    k_rot[0::2] = k[0::2] * cos_k - k[1::2] * sin_k
    k_rot[1::2] = k[0::2] * sin_k + k[1::2] * cos_k

    return float(np.dot(q_rot, k_rot))
