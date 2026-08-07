import numpy as np
from int4.packing import pack_int4

def quantize_weights(w, group_size=128):
    orig_shape = w.shape
    flat = w.reshape(-1, group_size)
    max_val = np.max(np.abs(flat), axis=1, keepdims=True)
    scale = max_val / 7.0
    scale = np.maximum(scale, 1e-8)
    quantized = np.round(flat / scale).astype(np.int8)
    quantized = np.clip(quantized, -8, 7)
    q_unsigned = (quantized + 8).astype(np.uint8)
    packed = pack_int4(q_unsigned)
    return packed, scale.astype(np.float32), orig_shape
