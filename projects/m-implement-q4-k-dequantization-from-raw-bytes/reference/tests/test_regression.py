import struct
from dequant import dequantize_q6_k

def test_q6_k_signed_scales():
    ql = bytes([0xFF] * 128)
    qh = bytes([0xFF] * 64)
    scales = bytes([0xFF] * 16)
    d = struct.pack('<e', 1.0)

    block = ql + qh + scales + d
    result = dequantize_q6_k(block)

    assert result[0] < 0, "Scale should be signed, output must be negative!"
    assert result[0] == -31.0
