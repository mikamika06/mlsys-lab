import numpy as np
from kvfp8.attn import compute_attention_error_by_position
from kvfp8.quant import dequantize_fp8_per_head, quantize_fp8_per_head


def test_per_head_quantization_is_required():
    rng = np.random.default_rng(42)
    seq_len, num_heads, head_dim = 32, 4, 64

    q = rng.normal(0, 1, size=(seq_len, num_heads, head_dim))
    k = rng.normal(0, 1, size=(seq_len, num_heads, head_dim))
    v = rng.normal(0, 1, size=(seq_len, num_heads, head_dim))

    k[:, 0, :] *= 100.0
    v[:, 0, :] *= 100.0

    err_per_head = compute_attention_error_by_position(
        q, k, v, quantize_fp8_per_head, dequantize_fp8_per_head
    )

    def global_quant(x):
        max_fp8 = 448.0
        scale = np.maximum(np.max(np.abs(x)) / max_fp8, 1e-12)
        q_val = np.clip(np.round(x / scale), -448.0, 448.0)
        return q_val, scale

    def global_dequant(q_val, scale):
        return q_val * scale

    err_global = compute_attention_error_by_position(
        q, k, v, global_quant, global_dequant
    )

    assert np.mean(err_per_head) < np.mean(err_global)
