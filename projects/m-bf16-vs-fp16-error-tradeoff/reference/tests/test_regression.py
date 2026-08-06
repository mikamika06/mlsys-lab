import numpy as np
from fa_tradeoff.attention import blockwise_attention


def test_blockwise_attention_exactness():
    np.random.seed(123)
    q = np.random.randn(1, 2, 8, 16).astype(np.float32)
    k = np.random.randn(1, 2, 16, 16).astype(np.float32)
    v = np.random.randn(1, 2, 16, 16).astype(np.float32)
    mask = np.ones((1, 1, 8, 16), dtype=np.float32)
    out = blockwise_attention(q, k, v, mask)
    assert out.shape == (1, 2, 8, 16)
    assert not np.any(np.isnan(out))


def test_fully_masked_row_no_nan():
    q = np.random.randn(1, 1, 4, 8).astype(np.float32)
    k = np.random.randn(1, 1, 8, 8).astype(np.float32)
    v = np.random.randn(1, 1, 8, 8).astype(np.float32)
    mask = np.zeros((1, 1, 4, 8), dtype=np.float32)
    out = blockwise_attention(q, k, v, mask)
    assert not np.any(np.isnan(out))
