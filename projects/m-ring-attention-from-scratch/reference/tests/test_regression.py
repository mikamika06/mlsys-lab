import sys
import numpy as np

sys.path.insert(0, ".")
from ringattn.crossover import compute_crossover
from ringattn.ring import ring_attention
from ringattn.ulysses import ulysses_reshuffle


def test_ring_shape():
    q = np.ones((1, 2, 4, 8))
    k = np.ones((1, 2, 4, 8))
    v = np.ones((1, 2, 4, 8))
    out = ring_attention(q, k, v, 2)
    assert out.shape == q.shape


def test_ulysses_shape():
    x = np.ones((1, 4, 4, 8))
    out = ulysses_reshuffle(x, 2)
    assert out.shape == (1, 4, 4, 8)


def test_crossover_values():
    res = compute_crossover(1024, 64, 4)
    assert res["ring"] > 0
    assert res["ulysses"] > 0
    assert res["better"] in ("ring", "ulysses")
