def fp4_quantize_scalar(val):
    """Quantize scalar or array values to nearest FP4 E2M1 code."""
    raise NotImplementedError


def nvfp4_quantize_dequantize(x, block_size=16, super_block_size=256):
    """Quantize and dequantize tensor using NVFP4 two-level scaling."""
    raise NotImplementedError


def mxfp4_quantize_dequantize(x, block_size=32):
    """Quantize and dequantize tensor using MXFP4 single-level scaling."""
    raise NotImplementedError
