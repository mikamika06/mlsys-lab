import numpy as np
from .fp8 import quantize_to_fp8_vals

def get_scales(x, q_max, axis=1):
    raise NotImplementedError

def quantize_int8(x, axis=1):
    raise NotImplementedError

def quantize_int4(x, axis=1):
    raise NotImplementedError

def quantize_fp8(x, axis=1):
    raise NotImplementedError
