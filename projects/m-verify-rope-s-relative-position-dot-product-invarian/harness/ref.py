import numpy as np


def compute_rope_frequencies(dim, max_seq_len, base=10000.0):
    theta = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    seq_idx = np.arange(max_seq_len, dtype=np.float64)
    idx_theta = np.outer(seq_idx, theta)
    return np.cos(idx_theta), np.sin(idx_theta)


def apply_rope(x, pos, freqs):
    cos, sin = freqs
    c = cos[pos]
    s = sin[pos]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    x_rot = np.empty_like(x)
    x_rot[..., 0::2] = x_even * c - x_odd * s
    x_rot[..., 1::2] = x_even * s + x_odd * c
    return x_rot


def rope_dot_product(q, k, m, n, freqs):
    q_rot = apply_rope(q, m, freqs)
    k_rot = apply_rope(k, n, freqs)
    return np.sum(q_rot * k_rot, axis=-1)


def apply_position_interpolation(pos, scale_factor):
    return pos / scale_factor


def compute_perplexity(logits, targets):
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_l = np.exp(shifted)
    probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
    N = targets.size
    p = probs.reshape(-1, probs.shape[-1])[np.arange(N), targets.reshape(-1)]
    return float(np.exp(np.mean(-np.log(np.clip(p, 1e-12, 1.0)))))


def generate_synthetic_data(dim=32, seq_len=1024, vocab_size=100, seed=42):
    rng = np.random.RandomState(seed)
    q = rng.randn(dim)
    k = rng.randn(dim)
    logits = rng.randn(seq_len, vocab_size)
    targets = rng.randint(0, vocab_size, size=(seq_len,))
    return q, k, logits, targets
