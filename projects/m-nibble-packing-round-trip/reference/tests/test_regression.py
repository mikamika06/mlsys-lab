import sys

sys.path.insert(0, ".")
from nibblepack import dequantize_block, pack_nibbles, unpack_nibbles

CASES = [
    [],
    [0],
    [15],
    [3, 10],
    [3, 10, 1, 8, 5],
    list(range(16)),
    [(i * 7 + 3) % 16 for i in range(17)],
]


def test_round_trip_preserves_codes():
    for codes in CASES:
        n = len(codes)
        packed = pack_nibbles(codes)
        back = [int(c) for c in unpack_nibbles(packed, n)]
        assert back == codes, f"n={n}: {back} != {codes}"


def test_packed_length_is_ceil_half():
    for codes in CASES:
        n = len(codes)
        packed = pack_nibbles(codes)
        assert len(packed) == (n + 1) // 2, f"n={n}: packed len {len(packed)}"


def test_dequantize_scales_linearly_with_absmax():
    codes = [0, 7, 15]
    packed = pack_nibbles(codes)
    a = dequantize_block(packed, 3, 2.0)
    b = dequantize_block(packed, 3, 4.0)
    for x, y in zip(a, b):
        assert abs(2 * float(x) - float(y)) < 1e-9, f"{x} * 2 != {y}"


def test_zero_absmax_gives_zero_block():
    codes = [1, 2, 3, 4, 5]
    packed = pack_nibbles(codes)
    out = dequantize_block(packed, len(codes), 0.0)
    assert all(float(v) == 0.0 for v in out), f"expected all zeros, got {list(out)}"
