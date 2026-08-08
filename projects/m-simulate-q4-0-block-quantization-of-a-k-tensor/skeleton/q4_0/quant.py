import numpy as np


def quantize(tensor):
    """
    Quantizes a tensor using Q4_0 block quantization (block size 32).
    Returns a dictionary with:
      - "shape": original shape of the tensor
      - "scales": float32 array of shape (num_blocks,)
      - "packed": uint8 array of shape (num_blocks, 16)
    """
    raise NotImplementedError


def dequantize(q_dict):
    """
    Dequantizes a Q4_0 dictionary back to a float32 tensor of the original shape.
    """
    raise NotImplementedError
