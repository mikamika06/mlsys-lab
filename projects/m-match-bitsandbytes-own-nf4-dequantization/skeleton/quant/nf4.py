import numpy as np

def get_nf4_table():
    """
    Return the 16-element NF4 quantile table as a float32 numpy array.
    """
    raise NotImplementedError

def unpack_indices(packed):
    """
    Unpack an array of uint8 into an array of uint8 twice the size.
    The high nibble forms the first element, the low nibble forms the second.
    """
    raise NotImplementedError

def dequantize(packed, absmax, block_size=64):
    """
    Dequantize packed bytes using the provided per-block absolute maximums.
    """
    raise NotImplementedError
