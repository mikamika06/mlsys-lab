import numpy as np
from moe_drops.router import compute_dropped_fraction

def test_detect_dropped_tokens():
    indices = np.array([0, 0, 0, 0, 1])
    drops = compute_dropped_fraction(indices, 2, 1.0)
    assert drops > 0.0
