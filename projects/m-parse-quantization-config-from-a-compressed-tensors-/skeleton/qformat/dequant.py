import numpy as np

def dequantize_w4a16(packed_weights, scales, zeros, group_size):
    """
    Dequantizes uint8 packed W4A16 weights into float32.
    """
    raise NotImplementedError

def dequantize_nvfp4(weights, local_scales, global_scale, group_size=16):
    """
    Dequantizes NVFP4 applying local and global scales.
    """
    raise NotImplementedError
