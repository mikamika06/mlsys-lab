import numpy as np
import sys
sys.path.insert(0, ".")
from adaptermerge.merge import merge_adapters
from adaptermerge.drift import compute_relative_error
from adaptermerge.evaluate import evaluate_drift

def test_merge_shape_and_linearity():
    rng = np.random.default_rng(42)
    w = rng.standard_normal((16, 16))
    d1 = rng.standard_normal((16, 16))
    d2 = rng.standard_normal((16, 16))
    res = merge_adapters(w, d1, d2, 0.5, 0.5)
    assert res.shape == (16, 16)
    expected = w + 0.5 * d1 + 0.5 * d2
    np.testing.assert_allclose(res, expected, rtol=1e-5, atol=1e-5)

def test_relative_error_bound():
    rng = np.random.default_rng(42)
    out_ref = rng.standard_normal((8, 16))
    out_merged = out_ref + 1e-7 * rng.standard_normal((8, 16))
    err = compute_relative_error(out_ref, out_merged)
    assert err < 1e-4

def test_evaluate_drift_zero():
    rng = np.random.default_rng(42)
    w = rng.standard_normal((16, 16))
    d1 = rng.standard_normal((16, 16))
    d2 = rng.standard_normal((16, 16))
    x = rng.standard_normal((4, 16))
    err = evaluate_drift(w, d1, d2, x, 1.0, 1.0)
    assert err < 1e-12
