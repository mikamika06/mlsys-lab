import numpy as np
from triton_ops.fuse import fuse_ops


def test_broadcasting_strides():
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([10.0, 20.0], dtype=np.float32)
    z = np.array([[2.0], [1.0]], dtype=np.float32)
    got = fuse_ops(x, y, z)
    want = (x + y) * z
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_identity_scaling():
    x = np.ones((16, 16), dtype=np.float32)
    y = np.zeros((16, 16), dtype=np.float32)
    z = np.ones((16, 16), dtype=np.float32)
    got = fuse_ops(x, y, z)
    np.testing.assert_allclose(got, x, rtol=1e-5, atol=1e-5)
