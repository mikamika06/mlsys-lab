import numpy as np


def calc_scale_zp_asymmetric(min_val, max_val, bits=8):
    raise NotImplementedError


def calc_scale_symmetric(max_abs_val, bits=8):
    raise NotImplementedError


def quantize_asymmetric(x, scale, zp, bits=8):
    raise NotImplementedError


def dequantize_asymmetric(x_q, scale, zp):
    raise NotImplementedError


def quantize_symmetric(x, scale, bits=8):
    raise NotImplementedError


def dequantize_symmetric(x_q, scale):
    raise NotImplementedError


def per_channel_weight_scales(w, bits=8):
    """
    w is assumed to be of shape (out_channels, in_channels, kernel_h, kernel_w)
    """
    raise NotImplementedError


def fused_requantize_scale(s_in, s_w, s_out):
    raise NotImplementedError
