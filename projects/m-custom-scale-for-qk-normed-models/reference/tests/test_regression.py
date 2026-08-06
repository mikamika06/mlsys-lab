import numpy as np
from qknorm.config import AttentionConfig
from qknorm.attention import compute_qknorm_attention


def test_qknorm_custom_scale():
    np.random.seed(42)
    b, h, s, d = 2, 4, 16, 32
    q = np.random.randn(b, h, s, d)
    k = np.random.randn(b, h, s, d)
    v = np.random.randn(b, h, s, d)

    cfg_default = AttentionConfig(head_dim=d, custom_scale=None)
    cfg_custom = AttentionConfig(head_dim=d, custom_scale=0.25)

    out_default = compute_qknorm_attention(q, k, v, cfg_default)
    out_custom = compute_qknorm_attention(q, k, v, cfg_custom)

    assert not np.allclose(out_default, out_custom), "Custom scale must alter attention output"
