import numpy as np

def pack_int4(w_q):
    flat = w_q.flatten()
    if len(flat) % 2 != 0:
        flat = np.pad(flat, (0, 1), mode='constant')
    even = flat[0::2].astype(np.uint8)
    odd = flat[1::2].astype(np.uint8)
    packed = (odd << 4) | (even & 0x0F)
    return packed

def unpack_int4(w_packed, shape):
    even = w_packed & 0x0F
    odd = (w_packed >> 4) & 0x0F
    unpacked = np.empty(w_packed.size * 2, dtype=np.uint8)
    unpacked[0::2] = even
    unpacked[1::2] = odd
    total_elements = np.prod(shape)
    unpacked = unpacked[:total_elements]
    quantized = unpacked.astype(np.int8) - 8
    return quantized.reshape(shape)
