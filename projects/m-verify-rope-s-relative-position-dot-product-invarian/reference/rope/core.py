import numpy as np


def compute_rope_frequencies(dim, max_seq_len, base=10000.0):
    """Compute sine and cosine frequencies for RoPE."""
    theta = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    seq_idx = np.arange(max_seq_len, dtype=np.float64)
    idx_theta = np.outer(seq_idx, theta)
    cos = np.cos(idx_theta)
    sin = np.sin(idx_theta)
    return cos, sin


def apply_rope(x, pos, freqs):
    """Apply rotary position embedding to vector x at position pos."""
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
    """Compute dot product between RoPE-transformed q at m and k at n."""
    q_rot = apply_rope(q, m, freqs)
    k_rot = apply_rope(k, n, freqs)
    return np.sum(q_rot * k_rot, axis=-1)


def apply_position_interpolation(pos, scale_factor):
    """Linearly rescale positions by scale_factor."""
    return pos / scale_factor


def compute_perplexity(logits, targets):
    """Compute perplexity given logits and integer targets."""
    shifted_logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    N = targets.size
    flat_probs = probs.reshape(-1, probs.shape[-1])
    flat_targets = targets.reshape(-1)
    correct_probs = flat_probs[np.arange(N), flat_targets]
    nll = -np.log(np.clip(correct_probs, 1e-12, 1.0))
    avg_nll = np.mean(nll)
    return float(np.exp(avg_nll))
