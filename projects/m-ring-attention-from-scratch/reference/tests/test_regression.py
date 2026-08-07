import sys

sys.path.insert(0, ".")
from ringattn.crossover import compute_crossover


def test_crossover_values_positive():
    ring_vol, ulysses_vol = compute_crossover(2048, 512, 4, 8)
    assert ring_vol > 0, "ring volume must be positive"
    assert ulysses_vol > 0, "ulysses volume must be positive"


def test_crossover_scaling():
    r1, u1 = compute_crossover(1024, 256, 2, 4)
    r2, u2 = compute_crossover(2048, 256, 2, 4)
    assert r2 > r1, "ring volume should scale with sequence length"
    assert u2 > u1, "ulysses volume should scale with sequence length"
