import sys
sys.path.insert(0, ".")
import numpy as np
from quant.scale import optimal_scale
from quant.round import weighted_round_to_nearest
from quant.gain import measure_gain


def test_optimal_scale_positive():
    w = np.array([1.5, -2.0, 3.1], dtype=np.float32)
    im = np.array([1.0, 2.0, 1.5], dtype=np.float32)
    s = optimal_scale(w, im, -8, 7)
    assert s > 0, f"scale {s} is not positive"


def test_weighted_round_shapes():
    w = np.random.default_rng(42).normal(size=64).astype(np.float32)
    im = np.random.default_rng(43).uniform(0.1, 5.0, size=64).astype(np.float32)
    s, q = weighted_round_to_nearest(w, im, -8, 7)
    assert q.shape == w.shape
    assert -8 <= q.min() and q.max() <= 7


def test_gain_across_bits():
    w = np.array([0.5, -1.2, 2.3, -0.8], dtype=np.float32)
    im = np.array([1.0, 1.0, 100.0, 100.0], dtype=np.float32)
    s_weighted = optimal_scale(w, im, -8, 7)
    ones = np.ones_like(im)
    s_unweighted = optimal_scale(w, ones, -8, 7)
    assert s_weighted != s_unweighted
