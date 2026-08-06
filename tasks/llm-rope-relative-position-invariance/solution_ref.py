import math

def rope_relative_dot(q: list[float], k: list[float], pos_q: int, pos_k: int) -> float:
    d = len(q)
    if d % 2 != 0:
        raise ValueError("q and k must have even length")
    q_rot = [0.0] * d
    k_rot = [0.0] * d
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
