import numpy as np

E2M1_LUT = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)

def dequantize_w4a16(packed_qweights, scales, zeros, group_size):
    N, K_packed = packed_qweights.shape
    K = K_packed * 2
    qweights = np.zeros((N, K), dtype=np.int32)
    qweights[:, 0::2] = packed_qweights & 0x0F
    qweights[:, 1::2] = (packed_qweights >> 4) & 0x0F
    groups_per_row = K // group_size
    qweights = qweights.reshape(N, groups_per_row, group_size)
    scales_expanded = scales[:, :, None]
    zeros_expanded = zeros[:, :, None]
    dequant = (qweights.astype(np.float32) - zeros_expanded.astype(np.float32)) * scales_expanded
    return dequant.reshape(N, K)

def dequantize_nvfp4(packed_fp4, local_scales, global_scale):
    N, K_packed = packed_fp4.shape
    K = K_packed * 2
    raw = np.zeros((N, K), dtype=np.uint8)
    raw[:, 0::2] = packed_fp4 & 0x0F
    raw[:, 1::2] = (packed_fp4 >> 4) & 0x0F
    sign = ((raw >> 3) & 0x01).astype(np.float32)
    sign = 1.0 - 2.0 * sign
    mag = raw & 0x07
    unscaled_vals = sign * E2M1_LUT[mag]
    group_size = 16
    groups_per_row = K // group_size
    unscaled_vals = unscaled_vals.reshape(N, groups_per_row, group_size)
    loc_scales = local_scales.astype(np.float32)[:, :, None]
    dequant = unscaled_vals * loc_scales * float(global_scale)
    return dequant.reshape(N, K)
