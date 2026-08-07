import numpy as np

def dequantize_w4a16(packed_weights, scales, zeros, group_size):
    N, K_half = packed_weights.shape
    K = K_half * 2
    w = np.empty((N, K), dtype=np.uint8)

    w[:, 0::2] = packed_weights & 0x0F
    w[:, 1::2] = (packed_weights >> 4) & 0x0F

    s = np.repeat(scales, group_size, axis=1)
    z = np.repeat(zeros, group_size, axis=1)

    return (w.astype(np.float32) - z) * s

def dequantize_nvfp4(weights, local_scales, global_scale, group_size=16):
    s = np.repeat(local_scales, group_size, axis=1)
    return weights * s * global_scale
