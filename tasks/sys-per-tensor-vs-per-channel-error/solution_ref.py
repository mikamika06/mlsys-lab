import numpy as np


def compare_quant_errors(W):
    W = np.asarray(W, dtype=np.float64)

    tensor_scale = np.max(np.abs(W)) / 127.0
    tensor_q = np.clip(np.round(W / tensor_scale), -127, 127)
    tensor_out = tensor_q * tensor_scale

    channel_scale = np.max(np.abs(W), axis=1, keepdims=True) / 127.0
    channel_q = np.clip(np.round(W / channel_scale), -127, 127)
    channel_out = channel_q * channel_scale

    return tensor_out, channel_out
