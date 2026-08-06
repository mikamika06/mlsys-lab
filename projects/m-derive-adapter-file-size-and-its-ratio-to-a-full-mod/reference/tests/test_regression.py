def test_regression():
    import numpy as np
    from adapter_merge.numerical import forward_equivalence

    np.random.seed(0)
    w = np.random.randn(32, 32).astype(np.float32)
    a = np.random.randn(4, 32).astype(np.float32)
    b = np.random.randn(32, 4).astype(np.float32)
    x = np.random.randn(8, 32).astype(np.float32)
    scale = 2.0

    err = forward_equivalence(x, w, a, b, scale)
    assert err < 1e-4, "Merged model forward pass is not mathematically equivalent"
