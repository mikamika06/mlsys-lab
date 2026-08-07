import numpy as np


def get_test_cases():
    np.random.seed(42)
    q = np.random.randn(1, 32, 64).astype(np.float32)
    k = np.random.randn(1, 128, 64).astype(np.float32)
    v = np.random.randn(1, 128, 64).astype(np.float32)
    return q, k, v


def naive_full_attention(q, k, v):
    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
    m = np.max(scores, axis=-1, keepdims=True)
    e = np.exp(scores - m)
    return np.matmul(e / np.sum(e, axis=-1, keepdims=True), v)
