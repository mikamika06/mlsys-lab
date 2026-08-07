import numpy as np
from int4.quant import quantize_weights

def get_test_weights(shape=(256, 256), seed=42):
    np.random.seed(seed)
    return np.random.randn(*shape).astype(np.float32)

def get_oracle_size(w):
    packed, scale, orig_shape = quantize_weights(w)
    return packed.nbytes + scale.nbytes

def get_oracle_reconstruction(w, group_size=128):
    packed, scale, orig_shape = quantize_weights(w, group_size)
    even = packed & 0x0F
    odd = (packed >> 4) & 0x0F
    unpacked = np.empty(packed.size * 2, dtype=np.uint8)
    unpacked[0::2] = even
    unpacked[1::2] = odd
    total_elements = np.prod(orig_shape)
    unpacked = unpacked[:total_elements]
    q = unpacked.astype(np.int8) - 8
    flat_q = q.reshape(-1, group_size)
    dequant = flat_q.astype(np.float32) * scale
    return dequant.reshape(orig_shape)
