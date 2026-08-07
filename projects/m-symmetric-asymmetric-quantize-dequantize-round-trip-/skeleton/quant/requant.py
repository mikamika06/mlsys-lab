import numpy as np


def compute_requant_scale(scale_input, scale_weight, scale_output):
    raise NotImplementedError


def requantize_int32(acc, scale_input, scale_weight, scale_output, zero_point_out=0, qmin=-128, qmax=127):
    raise NotImplementedError
