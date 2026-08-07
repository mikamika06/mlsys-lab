import numpy as np
from kvquant.quant import quantize_q8_0, dequantize_q8_0, max_abs_error_bound
from kvquant.attention import flash_attn_q8_0, unfused_attn_q8_0


def test_quantization_error_bounds():
    np.random.seed(42)
    data = np.random.uniform(-2.0, 2.0, size=(16, 64)).astype(np.float32)
    qdict = quantize_q8_0(data, block_size=32)
    rec = dequantize_q8_0(qdict)

    max_err = np.max(np.abs(data - rec))
    bound = max_abs_error_bound(data, block_size=32)

    assert max_err <= bound + 1e-6, f"Error {max_err} exceeded bound {bound}"


def test_flash_attention_required_for_quantized_kv():
    np.random.seed(42)
    q = np.random.randn(4, 64).astype(np.float32)
    k = np.random.randn(128, 64).astype(np.float32)
    v = np.random.randn(128, 64).astype(np.float32)

    k_qdict = quantize_q8_0(k, block_size=32)
    v_qdict = quantize_q8_0(v, block_size=32)

    out_flash = flash_attn_q8_0(q, k_qdict, v_qdict, sm_scale=0.125)
    out_unfused, mat_bytes = unfused_attn_q8_0(q, k_qdict, v_qdict, sm_scale=0.125)

    np.testing.assert_allclose(out_flash, out_unfused, atol=1e-4)

    quantized_kv_bytes = k_qdict["qdata"].nbytes + v_qdict["qdata"].nbytes
    assert mat_bytes > 2 * quantized_kv_bytes, (
        f"Materialized memory ({mat_bytes}B) should exceed quantized size ({quantized_kv_bytes}B)"
    )
