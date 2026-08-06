import numpy as np
from quant.blocks import decode_q4_block, encode_q4_block, find_optimal_distribution_params


def test_distribution_params_valid():
    res = find_optimal_distribution_params()
    assert "skew" in res
    assert isinstance(res["skew"], float)


def test_decode_encode_roundtrip():
    original = np.array([3, 12, 5, 9, 0, 15, 7, 8], dtype=np.uint8)
    encoded = encode_q4_block(original, "Q4_0")
    decoded = decode_q4_block(encoded, "Q4_0")
    np.testing.assert_array_equal(original, decoded)


def test_nibble_packing_order():
    raw_bytes = bytes([0x12, 0x34])
    decoded = decode_q4_block(raw_bytes, "Q4_0")
    np.testing.assert_array_equal(decoded, np.array([2, 1, 4, 3], dtype=np.uint8))
