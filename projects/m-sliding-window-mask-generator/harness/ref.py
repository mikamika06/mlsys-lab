import numpy as np


def generate_sliding_window_mask(seq_len, window_size):
    i = np.arange(seq_len)[:, None]
    j = np.arange(seq_len)[None, :]
    dist = i - j
    return (dist >= 0) & (dist < window_size)


def windowed_attention(q, k, v, mask):
    d_k = q.shape[-1]
    scores = np.matmul(q, k.swapaxes(-1, -2)) / np.sqrt(d_k)
    scores = np.where(mask, scores, -1e9)
    scores -= np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return np.matmul(probs, v)


def kv_cache_memory_bytes(seq_len, batch, layers, num_heads, d_head, dtype_bytes=2):
    total = 0
    for layer in layers:
        if layer["type"] == "global":
            total += seq_len
        elif layer["type"] == "sliding":
            total += min(seq_len, layer["window_size"])
    return total * batch * 2 * num_heads * d_head * dtype_bytes


def get_m1_fixtures():
    np.random.seed(42)
    q = np.random.randn(2, 4, 16, 32)
    k = np.random.randn(2, 4, 16, 32)
    v = np.random.randn(2, 4, 16, 32)
    return q, k, v


def get_m2_configs():
    return [
        (100, 2, [{"type": "global"}], 8, 128, 2),
        (500, 4, [{"type": "sliding", "window_size": 256}], 8, 128, 2),
        (8000, 1, [{"type": "global"}, {"type": "sliding", "window_size": 1024}], 32, 64, 2),
        (50, 1, [{"type": "sliding", "window_size": 1024}], 16, 64, 2)
    ]
