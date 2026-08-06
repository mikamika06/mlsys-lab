import numpy as np


def measure_yarn_attention_entropy(q, k, inv_freqs, attention_factor=1.0):
    seq_len, head_dim = q.shape
    t = np.arange(seq_len, dtype=np.float64)

    half_dim = head_dim // 2
    cos = np.cos(np.outer(t, inv_freqs[:half_dim]))
    sin = np.sin(np.outer(t, inv_freqs[:half_dim]))

    q_rot = np.zeros_like(q)
    k_rot = np.zeros_like(k)

    q_1, q_2 = q[:, :half_dim], q[:, half_dim:]
    k_1, k_2 = k[:, :half_dim], k[:, half_dim:]

    q_rot[:, :half_dim] = q_1 * cos - q_2 * sin
    q_rot[:, half_dim:] = q_1 * sin + q_2 * cos

    k_rot[:, :half_dim] = k_1 * cos - k_2 * sin
    k_rot[:, half_dim:] = k_1 * sin + k_2 * cos

    scores = (q_rot @ k_rot.T) / np.sqrt(head_dim)
    scores = scores * attention_factor

    mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    scores[mask] = -1e9

    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    exp_scores[mask] = 0.0

    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    entropy = -np.sum(attn_weights * np.log(attn_weights + 1e-12), axis=-1)
    return float(np.mean(entropy))
