import sys
import numpy as np

sys.path.insert(0, ".")
from ucp.fix import fix_merge

def test_fix_merge_preserves_shape_and_values():
    rng = np.random.default_rng(123)
    t = rng.standard_normal((100, 50))
    res = fix_merge(t, split_dim=0)
    assert res.shape == (100, 50)
    assert np.allclose(res, t)

def test_fix_merge_identity_on_swapped_axes_detection():
    rng = np.random.default_rng(456)
    t = rng.standard_normal((100, 50))
    res = fix_merge(t, split_dim=0)
    assert not np.array_equal(res, np.swapaxes(t, 0, 1))
