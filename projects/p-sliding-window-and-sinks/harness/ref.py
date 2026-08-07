import numpy as np


def generate_synthetic_data(seq_len: int, head_dim: int, seed: int = 42):
    rng = np.random.RandomState(seed)
    q = rng.randn(seq_len, head_dim)
    k = rng.randn(seq_len, head_dim)
    v = rng.randn(seq_len, head_dim)
    return q, k, v


def full_causal_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    L, d = q.shape
    scale = 1.0 / np.sqrt(d)
    out = np.zeros_like(q)
    for i in range(L):
        qi = q[i : i + 1]
        ki = k[: i + 1]
        vi = v[: i + 1]
        scores = np.dot(qi, ki.T) * scale
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_s = np.exp(scores - scores_max)
        w = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        out[i] = np.dot(w, vi)[0]
    return out


def naive_sliding_window_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, window_size: int
) -> np.ndarray:
    L, d = q.shape
    scale = 1.0 / np.sqrt(d)
    out = np.zeros_like(q)
    for i in range(L):
        start = max(0, i - window_size + 1)
        qi = q[i : i + 1]
        ki = k[start : i + 1]
        vi = v[start : i + 1]
        scores = np.dot(qi, ki.T) * scale
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_s = np.exp(scores - scores_max)
        w = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        out[i] = np.dot(w, vi)[0]
    return out


def generate_needle_in_haystack(
    seq_len: int, head_dim: int, needle_pos: int, seed: int = 123
):
    rng = np.random.RandomState(seed)
    q = rng.randn(seq_len, head_dim) * 0.1
    k = rng.randn(seq_len, head_dim) * 0.1
    v = rng.randn(seq_len, head_dim) * 0.1

    sink_key = rng.randn(head_dim)
    sink_key = sink_key / np.linalg.norm(sink_key) * 5.0
    k[0] = sink_key

    needle_key = rng.randn(head_dim)
    needle_key = needle_key / np.linalg.norm(needle_key) * 10.0
    needle_val = rng.randn(head_dim) * 2.0

    k[needle_pos] = needle_key
    v[needle_pos] = needle_val

    q[-1] = needle_key
    return q, k, v, needle_val
