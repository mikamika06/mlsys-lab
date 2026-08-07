import sys
import numpy as np

sys.path.insert(0, ".")
from fp8kv.quant import get_per_head_scale


def test_per_head_scale_isolates_outliers():
    x = np.ones((10, 4, 16))
    x[:, 0, :] = 1000.0
    scales = get_per_head_scale(x)
    assert scales.shape == (1, 4, 1)
    assert scales[0, 0, 0] < scales[0, 1, 0]
