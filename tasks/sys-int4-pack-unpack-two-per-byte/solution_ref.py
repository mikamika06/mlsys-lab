import numpy as np

def pack_int4(values: np.ndarray) -> np.ndarray:
    n = values.size
    padded_len = n + (n % 2)
    padded = np.zeros(padded_len, dtype=np.uint8)   # zeros, not empty: the pad nibble must be defined
    padded[:n] = values
    if n % 2:
        padded[-1] &= 0xF0   # keep high nibble only
    high = (padded[::2] << 4) & 0xF0
    low = padded[1::2]
    packed = high | low
    return packed

def unpack_int4(packed: np.ndarray, length: int) -> np.ndarray:
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    unpacked = np.zeros(length + (length % 2), dtype=np.uint8)
    unpacked[::2] = high[:len(high)]
    unpacked[1::2] = low[:len(low)]
    return unpacked[:length]
