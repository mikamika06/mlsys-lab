import sys
import numpy as np

sys.path.insert(0, ".")
from smoothquant.scale import compute_migration_scales, apply_smoothquant
from smoothquant.autotune import sweep_alpha_per_layer


def test_smoothquant_equivalence():
    np.random.seed(42)
    X = np.random.randn(16, 64).astype(np.float32)
    W = np.random.randn(32, 64).astype(np.float32)
    act_max = np.max(np.abs(X), axis=0)
    weight_max = np.max(np.abs(W), axis=0)

    s = compute_migration_scales(act_max, weight_max, 0.5)
    X_s, W_s = apply_smoothquant(X, W, s)

    orig_out = X @ W.T
    smooth_out = X_s @ W_s.T
    np.testing.assert_allclose(orig_out, smooth_out, rtol=1e-4, atol=1e-4)


def test_autotune_reduces_mse():
    np.random.seed(42)
    X = np.random.randn(16, 32).astype(np.float32)
    X[:, 0] *= 50.0
    W = np.random.randn(16, 32).astype(np.float32)

    candidates = [0.0, 0.5, 0.95]
    res = sweep_alpha_per_layer({"layer1": X}, {"layer1": W}, candidates)
    best = res["layer1"]["alpha"]

    assert best in candidates
    assert best != 0.0
    assert res["layer1"]["mse"] < 1000.0
