import numpy as np


def simulate_e4m3(x, scale):
    """
    Simulate E4M3 quantization by scaling, rounding, clipping to [-448.0, 448.0],
    and dequantizing.
    """
    raise NotImplementedError


def get_per_tensor_scale(x, max_val=448.0):
    """
    Return a single float scale factor for the entire tensor.
    If the tensor is all zeros, return 1.0.
    """
    raise NotImplementedError


def get_per_head_scale(x, max_val=448.0):
    """
    x has shape (seq_len, num_heads, head_dim).
    Return a scale factor per head, reshaped to (1, num_heads, 1).
    Heads that are all zeros should get a scale of 1.0.
    """
    raise NotImplementedError
