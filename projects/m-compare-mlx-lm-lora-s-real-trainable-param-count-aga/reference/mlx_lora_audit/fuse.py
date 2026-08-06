import numpy as np


def verify_fusion(base_layer, lora_a, lora_b, scale, fused_layer, use_dora=False, magnitude_vector=None):
    """Verify that fused layer weights match unfused base + adapter weights."""
    delta_w = (lora_b @ lora_a) * scale
    if not use_dora:
        expected_fused = base_layer + delta_w
    else:
        w_comb = base_layer + delta_w
        norm_w = np.linalg.norm(w_comb, axis=1, keepdims=True)
        norm_w = np.where(norm_w == 0, 1.0, norm_w)
        if magnitude_vector is None:
            magnitude_vector = np.linalg.norm(base_layer, axis=1, keepdims=True)
        else:
            magnitude_vector = np.reshape(magnitude_vector, (-1, 1))
        expected_fused = magnitude_vector * (w_comb / norm_w)

    max_diff = float(np.max(np.abs(fused_layer - expected_fused)))
    is_equivalent = max_diff < 1e-5
    return {
        "max_diff": max_diff,
        "is_equivalent": is_equivalent,
        "expected_fused": expected_fused
    }
