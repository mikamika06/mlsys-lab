import numpy as np


def derive_bias_scale(input_scale, weight_scale):
    if isinstance(weight_scale, np.ndarray):
        return input_scale * weight_scale
    return float(input_scale * weight_scale)


def dequantize_weights(weight_int8, weight_scale, weight_zero_point):
    w_f = weight_int8.astype(np.float32) - weight_zero_point.astype(np.float32)
    if isinstance(weight_scale, np.ndarray):
        shape = (-1,) + (1,) * (w_f.ndim - 1)
        scale_broadcast = weight_scale.reshape(shape)
        return w_f * scale_broadcast
    return w_f * weight_scale


def compute_quantized_conv(input_int8, weight_int8, bias_int32, input_scale, weight_scale, out_scale, out_zero_point):
    accum = np.sum(
        input_int8.astype(np.int32)[..., None, :, :] * weight_int8.astype(np.int32)[:, :, None, :],
        axis=(1, 2, 3)
    )
    if bias_int32 is not None:
        accum = accum + bias_int32.astype(np.int32)
    bias_scale = derive_bias_scale(input_scale, weight_scale)
    if isinstance(bias_scale, np.ndarray):
        shape = (-1,) + (1,) * (accum.ndim - 1)
        bias_scale_b = bias_scale.reshape(shape)
    else:
        bias_scale_b = bias_scale
    float_output = accum.astype(np.float32) * bias_scale_b
    quantized_output = np.round(float_output / out_scale) + out_zero_point
    return np.clip(quantized_output, -128, 127).astype(np.int8)
