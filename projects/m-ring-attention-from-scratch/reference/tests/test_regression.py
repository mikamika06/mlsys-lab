import sys
import numpy as np

sys.path.insert(0, ".")
from ringattn.ring import ring_attention
from ringattn.ulysses import ulysses_reshuffle
from ringattn.crossover import compute_crossover

def test_ring_attention_output_shape():
    q = np.ones((1, 16, 2, 8), dtype=np.float32)
    k = np.ones((1, 16, 2, 8), dtype=np.float32)
    v = np.ones((1, 16, 2, 8), dtype=np.float32)
    out = ring_attention(q, k, v, 2)
    assert out.shape == q.shape

def test_ulysses_reshuffle_shape():
    x = np.ones((1, 8, 4, 8), dtype=np.float32)
    out = ulysses_reshuffle(x, 2, 0, forward=True)
    assert out.shape == (1, 8, 2, 8)

def test_crossover_monotonicity():
    r1, u1 = compute_crossover(1024, 4096, 8)
    r2, u2 = compute_crossover(8192, 4096, 8)
    assert r2 > r1
    assert u2 > u1
    assert u1 < r1
