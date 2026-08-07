import numpy as np


def ring_attention(q, k, v, steps):
    """Compute ring attention reference."""
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


def ulysses_reshuffle(x, world_size):
    """Perform Ulysses All-to-All reshuffle reference."""
    b, s, h, d = x.shape
    s_sub = s // world_size
    h_sub = h // world_size
    reshaped = x.reshape(b, s_sub, world_size, h_sub, world_size, d)
    transposed = np.transpose(reshaped, (0, 4, 1, 2, 3, 5))
    return transposed.reshape(b, world_size * s_sub, h, d)


def compute_crossover(seq_len, hidden_dim, world_size):
    """Compute communication volume for ring vs ulysses reference."""
    ring_vol = 2 * (world_size - 1) * seq_len * hidden_dim
    ulysses_vol = 2 * (world_size - 1) * seq_len * hidden_dim / (world_size * world_size)
    return {
        "ring": float(ring_vol),
        "ulysses": float(ulysses_vol),
        "better": "ulysses" if ulysses_vol < ring_vol else "ring"
    }
