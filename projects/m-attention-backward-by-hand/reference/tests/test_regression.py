import numpy as np
from attnbwd.backward import attention_backward, generate_dropout_mask


def test_dropout_reproducibility():
    shape = (2, 4, 16, 16)
    mask1 = generate_dropout_mask(shape, 0.2, 42)
    mask2 = generate_dropout_mask(shape, 0.2, 42)
    assert np.array_equal(mask1, mask2)


def test_backward_determinism():
    Q = np.random.randn(2, 2, 8, 16)
    K = np.random.randn(2, 2, 8, 16)
    V = np.random.randn(2, 2, 8, 16)
    dO = np.random.randn(2, 2, 8, 16)
    dQ1, dK1, dV1 = attention_backward(Q, K, V, dO, 0.2, 42)
    dQ2, dK2, dV2 = attention_backward(Q, K, V, dO, 0.2, 42)
    assert np.allclose(dQ1, dQ2)
    assert np.allclose(dK1, dK2)
    assert np.allclose(dV1, dV2)
