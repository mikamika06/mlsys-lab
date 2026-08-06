import numpy as np
from awq.scale import fold_scales, quantize_per_tensor


def test_scale_folding_invariant():
    np.random.seed(42)
    X = np.random.randn(16, 32)
    W = np.random.randn(32, 64)
    scales = np.random.uniform(0.5, 2.0, size=32)
    
    Y_ref = X @ W
    X_s, W_s = fold_scales(X, W, scales)
    Y_folded = X_s @ W_s
    
    np.testing.assert_allclose(Y_folded, Y_ref, rtol=1e-5, atol=1e-5)
