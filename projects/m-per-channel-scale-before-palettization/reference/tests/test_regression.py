import sys

sys.path.insert(0, ".")
import numpy as np
from palettize.decode import decode_weight
from palettize.scale import per_channel_scale
from palettize.sweep import pareto_sweep


def test_decode_reconstruction():
    w = np.random.randn(16, 16).astype(np.float32)
    q, s, z = per_channel_scale(w, 4)
    dec = decode_weight(q, s, z)
    assert dec.shape == w.shape
    assert np.mean((w - dec) ** 2) < 0.1


def test_pareto_monotonicity():
    w = np.random.randn(16, 16).astype(np.float32)
    res = pareto_sweep(w, [2, 4, 8])
    mses = [r["mse"] for r in res]
    assert mses[0] >= mses[1] >= mses[2]
