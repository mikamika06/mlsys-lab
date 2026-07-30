import numpy as np


def pack_nibbles(codes):
    codes = np.asarray(codes, dtype=np.uint8).reshape(-1)
    n = codes.shape[0]
    nbytes = (n + 1) // 2
    packed = np.zeros(nbytes, dtype=np.uint8)
    lo = codes[0::2] & np.uint8(0x0F)
    packed[: lo.shape[0]] |= lo
    hi = codes[1::2] & np.uint8(0x0F)
    packed[: hi.shape[0]] |= (hi << np.uint8(4))
    return packed


def unpack_nibbles(packed, n):
    packed = np.asarray(packed, dtype=np.uint8).reshape(-1)
    codes = np.zeros(n, dtype=np.uint8)
    n_even = (n + 1) // 2
    n_odd = n // 2
    codes[0::2] = packed[:n_even] & np.uint8(0x0F)
    codes[1::2] = (packed[:n_odd] >> np.uint8(4)) & np.uint8(0x0F)
    return codes
