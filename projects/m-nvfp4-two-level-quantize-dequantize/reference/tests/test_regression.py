import sys
import numpy as np

sys.path.insert(0, ".")
from quant.formats import nvfp4


def test_nvfp4_preserves_small_values_in_mixed_superblock():
    x = np.zeros(256)
    x[:16] = 6.0
    x[16:32] = 0.09

    q = nvfp4(x)
    assert np.any(np.abs(q[16:32]) > 0.0), "Small values vanished, vector scale is broken"
