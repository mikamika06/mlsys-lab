import numpy as np


def ring_attention(q, k, v, steps):
    """Compute ring attention."""
    b, h, s, d = q.shape
    out = np.zeros_like(q)
    k_ring = k.copy()
    v_ring = v.copy()
    for _ in range(steps):
        scores = np.matmul(q, np.swapaxes(k_ring, -1, -2)) / np.sqrt(d)
        max_s = np.max(scores, axis=-1, keepdims=True)
        exp_s = np.exp(scores - max_s)
        sum_exp = np.sum(exp_s, axis=-1, keepdims=True)
        out += np.matmul(exp_s, v_ring) / sum_exp
        k_ring = np.roll(k_ring, shift=1, axis=-2)
        v_ring = np.roll(v_ring, shift=1, axis=-2)
    return out / steps
