import sys
import numpy as np
sys.path.insert(0, ".")

from ring.imbalance import analyze_naive_ring
from ring.simulate import ring_attention_simulate

def test_imbalance_total_blocks():
    res = analyze_naive_ring(4)
    for row in res:
        assert row["fully_unmasked"] + row["partially_unmasked"] + row["fully_masked"] == 4

def test_simulate_causality():
    np.random.seed(42)
    q = [np.random.randn(4, 8) for _ in range(3)]
    k = [np.random.randn(4, 8) for _ in range(3)]
    v = [np.random.randn(4, 8) for _ in range(3)]

    out = ring_attention_simulate(q, k, v)

    Q = np.concatenate(q, axis=0)
    K = np.concatenate(k, axis=0)
    V = np.concatenate(v, axis=0)

    scores = Q @ K.T
    mask = np.triu(np.ones_like(scores, dtype=bool), k=1)
    scores[mask] = -np.inf

    m = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - m)
    P = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    O = P @ V

    O_shards = np.split(O, 3, axis=0)

    for i in range(3):
        np.testing.assert_allclose(out[i], O_shards[i], atol=1e-5)
