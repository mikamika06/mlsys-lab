def dequantize_w4a16(packed_qweights, scales, zeros, group_size):
    """Dequantize uint8 packed W4A16 group quantized weights."""
    raise NotImplementedError

def dequantize_nvfp4(packed_fp4, local_scales, global_scale):
    """Dequantize uint8 packed FP4 weights using local and global scales."""
    raise NotImplementedError
