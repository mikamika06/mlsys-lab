import numpy as np
from prune.imp import apply_mask, magnitude_mask


def test_mask_application():
    w = np.array([1.0, -2.0, 3.0])
    m = np.array([1.0, 0.0, 1.0])
    res = apply_mask(w, m)
    assert np.allclose(res, [1.0, 0.0, 3.0])


def test_sparsity_level():
    w = np.array([1.0, 2.0, 3.0, 4.0])
    m = magnitude_mask(w, 0.5)
    assert np.isclose(np.mean(m == 0.0), 0.5)
