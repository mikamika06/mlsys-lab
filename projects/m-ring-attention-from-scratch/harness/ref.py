import numpy as np
from reference.ringattn.core import ring_attention as ref_ring
from reference.ringattn.ulysses import ulysses_attention as ref_ulysses
from reference.ringattn.crossover import compute_crossover as ref_crossover


def generate_inputs():
    np.random.seed(123)
    q = np.random.randn(32, 64)
    k = np.random.randn(32, 64)
    v = np.random.randn(32, 64)
    return q, k, v


def generate_ulysses_inputs():
    np.random.seed(456)
    q = np.random.randn(1, 32, 64)
    k = np.random.randn(1, 32, 64)
    v = np.random.randn(1, 32, 64)
    return q, k, v
