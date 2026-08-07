import numpy as np
from ringattn.devices import ring_attention
from ringattn.memory import crossover_point


def test_ring_matches_full_attention():
    np.random.seed(0)
    q = np.random.randn(1, 16, 32).astype(np.float32)
    k = np.random.randn(1, 64, 32).astype(np.float32)
    v = np.random.randn(1, 64, 32).astype(np.float32)
    out_ring = ring_attention(q, k, v, num_devices=4)

    scale = 1.0 / np.sqrt(32)
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
    e_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    out_full = np.matmul(e_scores / np.sum(e_scores, axis=-1, keepdims=True), v)

    assert np.max(np.abs(out_ring - out_full)) < 1e-4


def test_crossover_positive():
    res = crossover_point(4096, 512, 8, 4)
    assert res > 0
