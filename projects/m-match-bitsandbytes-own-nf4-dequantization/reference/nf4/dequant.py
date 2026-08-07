import numpy as np


def get_nf4_table():
    return np.array([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
    ], dtype=np.float32)


def unpack_4bit(packed_bytes):
    unpacked = np.empty(len(packed_bytes) * 2, dtype=np.uint8)
    unpacked[0::2] = packed_bytes & 0x0F
    unpacked[1::2] = packed_bytes >> 4
    return unpacked


def dequantize_nf4(packed_bytes, absmax, blocksize=64):
    unpacked = unpack_4bit(packed_bytes)
    table = get_nf4_table()
    dequantized = table[unpacked]
    expanded_absmax = np.repeat(absmax, blocksize)
    return dequantized * expanded_absmax
