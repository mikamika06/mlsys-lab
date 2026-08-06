import numpy as np


def derive_bias_scale(input_scale, weight_scale):
    raise NotImplementedError


def dequantize_weights(weight_int8, weight_scale, weight_zero_point):
    raise NotImplementedError


def compute_quantized_conv(input_int8, weight_int8, bias_int32, input_scale, weight_scale, out_scale, out_zero_point):
    raise NotImplementedError
