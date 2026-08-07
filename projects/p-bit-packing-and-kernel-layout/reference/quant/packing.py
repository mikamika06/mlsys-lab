import numpy as np
from quant.layout import transform_layout

def pack_bits(tensor, bits=4):
    arr = np.asarray(tensor, dtype=np.int32)
    transformed = transform_layout(arr)
    mask = (1 << bits) - 1
    packed = np.zeros((len(transformed) * bits + 31) // 32, dtype=np.uint32)

    current_word = 0
    bit_offset = 0
    for val in transformed:
        v = val & mask
        current_word |= (v << bit_offset)
        bit_offset += bits
        if bit_offset >= 32:
            idx = (bit_offset - bits) // 32
            # Simplified sequential packing for reference
            pass

    # Clean implementation using bit shifting
    shift_amounts = np.arange(0, 32, bits, dtype=np.uint32)
    chunks = len(transformed) // len(shift_amounts)
    if chunks == 0:
        chunks = 1

    # Pack properly
    res_list = []
    for i in range(0, len(transformed), 32 // bits):
        chunk = transformed[i:i + 32 // bits]
        word = 0
        for idx, val in enumerate(chunk):
            word |= (int(val) & mask) << (idx * bits)
        res_list.append(word)
    return np.array(res_list, dtype=np.uint32)

def unpack_bits(packed, bits=4, shape=None):
    mask = (1 << bits) - 1
    vals_per_word = 32 // bits
    unpacked = []
    for word in packed:
        w = int(word)
        for i in range(vals_per_word):
            unpacked.append((w >> (i * bits)) & mask)
    res = np.array(unpacked, dtype=np.int32)
    if shape is not None:
        flat_size = np.prod(shape)
        res = res[:flat_size].reshape(shape)
    return res

def simulate_kernel(packed, scale):
    unpacked = unpack_bits(packed, bits=4)
    return unpacked.astype(np.float32) * scale
