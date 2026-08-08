import sys
sys.path.insert(0, ".")
import numpy as np
from tpp.parallel import column_parallel_matmul, row_parallel_matmul, tp_communication_volume, dtensor_mlp


def test_column_parallel_basic():
    x = np.ones((2, 4, 8))
    w = np.ones((8, 16))
    out = column_parallel_matmul(x, w)
    assert out.shape == (2, 4, 16)


def test_row_parallel_basic():
    x = np.ones((2, 4, 8))
    w = np.ones((8, 16))
    out = row_parallel_matmul(x, w)
    assert out.shape == (2, 4, 16)


def test_communication_volume_scaling():
    vol = tp_communication_volume(1, 128, 768, 3072, 4)
    assert vol["total_bytes"] > 0
    assert vol["forward_bytes"] == vol["backward_bytes"]


def test_dtensor_mlp_shape():
    np.random.seed(42)
    x = np.random.randn(1, 16, 32)
    w1 = np.random.randn(32, 64)
    w2 = np.random.randn(64, 32)
    out = dtensor_mlp(x, w1, w2, 2)
    assert out.shape == (1, 16, 32)
