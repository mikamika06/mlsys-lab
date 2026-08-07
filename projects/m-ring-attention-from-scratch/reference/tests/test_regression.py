import sys
import numpy as np

sys.path.insert(0, ".")
from ringattn.core import ring_attention
from ringattn.ulysses import ulysses_attention
from ringattn.crossover import compute_crossover


def test_ring_attention_basic():
    np.random.seed(42)
    q = np.random.randn(16, 32)
    k = np.random.randn(16, 32)
    v = np.random.randn(16, 32)
    out = ring_attention(q, k, v, 0, 2)
    assert out.shape == (16, 32)


def test_ulysses_attention_basic():
    np.random.seed(42)
    q = np.random.randn(1, 16, 32)
    k = np.random.randn(1, 16, 32)
    v = np.random.randn(1, 16, 32)
    out = ulysses_attention(q, k, v, 0, 2, 4)
    assert out.shape == (1, 16, 16)


def test_compute_crossover_monotonicity():
    seqs = [1024, 2048, 4096, 8192]
    res = compute_crossover(seqs, 4096, 8, 100.0, 1.0)
    assert len(res) == len(seqs)
