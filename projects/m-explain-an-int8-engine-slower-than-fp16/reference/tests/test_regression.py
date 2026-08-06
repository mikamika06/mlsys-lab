import sys
import numpy as np

sys.path.insert(0, ".")
from int8opt.scales import recover_per_channel_scales


def test_recover_scales_shape():
    mock = {"scales": np.array([1.0, 2.0], dtype=np.float32), "nodes": []}
    scales = recover_per_channel_scales(mock)
    assert isinstance(scales, np.ndarray)
    assert scales.shape[0] == 2


def test_recover_scales_values():
    mock = {"scales": np.array([0.5, 1.5], dtype=np.float32), "nodes": []}
    scales = recover_per_channel_scales(mock)
    assert np.allclose(scales, [0.5, 1.5])
