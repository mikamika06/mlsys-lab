import numpy as np


def quantize_symmetric(x, qmin=-128, qmax=127):
    raise NotImplementedError


def dequantize_symmetric(q, scale):
    raise NotImplementedError


def quantize_asymmetric(x, qmin=0, qmax=255):
    raise NotImplementedError


def dequantize_asymmetric(q, scale, zero_point, qmin=0, qmax=255):
    raise NotImplementedError
