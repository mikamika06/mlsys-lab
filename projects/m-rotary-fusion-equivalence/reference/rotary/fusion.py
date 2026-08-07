import numpy as np

def apply_fused_rotary(q, k, cos, sin):
    d = q.shape[-1]
    q1 = q[..., :d//2]
    q2 = q[..., d//2:]
    k1 = k[..., :d//2]
    k2 = k[..., d//2:]

    q_out = np.concatenate([q1 * cos - q2 * sin, q2 * cos + q1 * sin], axis=-1)
    k_out = np.concatenate([k1 * cos - k2 * sin, k2 * cos + k1 * sin], axis=-1)
    return q_out, k_out
