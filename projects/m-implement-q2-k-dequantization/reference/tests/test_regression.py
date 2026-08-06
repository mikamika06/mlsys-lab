import numpy as np
from kquant.dequant import dequantize_q2_k, reconstruct_q3_k_scales
from kquant.metrics import calculate_kquant_bpw


def test_q2_k_dequantization_accuracy():
    scales = np.array([0x21] * 16, dtype=np.uint8)
    qs = np.array([0b11100100] * 64, dtype=np.uint8)
    d = np.float16(2.0)
    dmin = np.float16(1.0)

    raw = scales.tobytes() + qs.tobytes() + d.tobytes() + dmin.tobytes()
    out = dequantize_q2_k(raw)

    assert out.shape == (256,)
    assert np.allclose(out[0:16], 2.0 * 0 - 2.0 * 1)


def test_q3_k_hmask_reconstruction():
    hmask = bytes([0b00010000] * 8)
    scales_raw = bytes([0x05] * 16)
    res = reconstruct_q3_k_scales(hmask, scales_raw)
    assert res.shape == (16,)
    assert res[0] == (5 - 32)
    assert res[1] == ((5 | (1 << 4)) - 32)


def test_kquant_bpw_values():
    assert abs(calculate_kquant_bpw("Q2_K") - 2.5) < 1e-6
    assert abs(calculate_kquant_bpw("Q3_K") - 3.4375) < 1e-6
