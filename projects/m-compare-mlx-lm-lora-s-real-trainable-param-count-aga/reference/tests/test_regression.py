import numpy as np
from mlx_lora_audit.fuse import verify_fusion


def test_fusion_verification():
    """Verify that fusion verification detects incorrect adapter fusion implementations."""
    np.random.seed(123)
    in_dim, out_dim, r = 16, 32, 4
    scale = 0.5
    base_layer = np.random.randn(out_dim, in_dim).astype(np.float32)
    lora_a = np.random.randn(r, in_dim).astype(np.float32)
    lora_b = np.random.randn(out_dim, r).astype(np.float32)

    fused_correct = base_layer + (lora_b @ lora_a) * scale
    res_good = verify_fusion(base_layer, lora_a, lora_b, scale, fused_correct)
    assert res_good["is_equivalent"], "Should accept correctly fused weights"

    fused_bad = base_layer + (lora_b @ lora_a)
    res_bad = verify_fusion(base_layer, lora_a, lora_b, scale, fused_bad)
    assert not res_bad["is_equivalent"], "Should reject unscaled fusion"
