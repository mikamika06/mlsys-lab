import sys
sys.path.insert(0, ".")
from metalops.gelu import custom_gelu
from metalops.matmul import custom_matmul
from metalops.tune import tune_threadgroup
import numpy as np


def test_gelu_output_shape_and_values():
    x = np.ones((10, 10), dtype=np.float32)
    out = custom_gelu(x)
    assert out.shape == (10, 10)
    assert not np.isnan(out).any()
    assert np.all(out > 0.0)


def test_matmul_identity():
    a = np.eye(32, dtype=np.float32)
    b = np.ones((32, 32), dtype=np.float32)
    out = custom_matmul(a, b)
    assert np.allclose(out, b)


def test_tune_threadgroup_valid():
    tg = tune_threadgroup((32, 32))
    assert isinstance(tg, tuple)
    assert len(tg) in (2, 3)
    assert np.prod(tg[:2]) <= 1024
