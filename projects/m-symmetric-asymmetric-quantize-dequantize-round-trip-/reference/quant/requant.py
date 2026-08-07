import numpy as np


def compute_requant_scale(scale_input, scale_weight, scale_output):
    effective_scale = (scale_input * scale_weight) / scale_output
    return effective_scale


def requantize_int32(acc, scale_input, scale_weight, scale_output, zero_point_out=0, qmin=-128, qmax=127):
    m = compute_requant_scale(scale_input, scale_weight, scale_output)
    scaled = acc.astype(np.float32) * m + zero_point_out
    q = np.clip(np.round(scaled), qmin, qmax).astype(np.int32)
    return q
