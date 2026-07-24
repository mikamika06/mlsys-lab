import numpy as np


def dequantize_uint4(codes, scale, zero_point):
    """Dequantize uint4 codes using float-domain zero-point convention."""
    return (np.asarray(codes, dtype=np.float32) * scale + zero_point).astype(np.float32)
