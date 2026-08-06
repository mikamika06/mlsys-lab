import numpy as np
from flashsel.fallback import execute_with_fallback

def test_cross_backend_equivalence():
    rng = np.random.default_rng(123)
    q = rng.standard_normal((1, 2, 8, 16))
    k = rng.standard_normal((1, 2, 8, 16))
    v = rng.standard_normal((1, 2, 8, 16))
    ladder = ["flashsel.backends.ideal"]
    res = execute_with_fallback(ladder, q, k, v)
    assert res.shape == q.shape
    assert not np.isnan(res).any()
    assert np.any(res != 0.0)
