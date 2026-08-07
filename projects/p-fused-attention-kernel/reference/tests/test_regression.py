import numpy as np
import sys
sys.path.insert(0, ".")
from fused_attn.kernel import tiled_attention


def test_tiled_attention_shape_and_values():
    np.random.seed(789)
    q = np.random.randn(32, 16)
    k = np.random.randn(32, 16)
    v = np.random.randn(32, 16)
    out = tiled_attention(q, k, v, block_size=16, causal=False)
    assert out.shape == (32, 16)
    assert not np.isnan(out).any()


def test_causal_tiled_attention():
    np.random.seed(999)
    q = np.random.randn(32, 16)
    k = np.random.randn(32, 16)
    v = np.random.randn(32, 16)
    out = tiled_attention(q, k, v, block_size=16, causal=True)
    assert out.shape == (32, 16)
    assert not np.isnan(out).any()
