import numpy as np


def unpack_mxfp4(packed_bytes: np.ndarray) -> np.ndarray:
    packed = np.asarray(packed_bytes, dtype=np.uint8)
    low_nibbles = packed & 0x0F
    high_nibbles = (packed >> 4) & 0x0F
    unpacked = np.empty((packed.size * 2,), dtype=np.uint8)
    unpacked[0::2] = low_nibbles.ravel()
    unpacked[1::2] = high_nibbles.ravel()
    return unpacked.reshape(packed.shape[:-1] + (-1,))
