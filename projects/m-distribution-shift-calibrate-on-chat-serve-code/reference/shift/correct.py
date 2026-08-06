import numpy as np


def calibrate_for_code(weights, chat_scales, code_sample):
    code_scales = np.max(np.abs(code_sample), axis=0, keepdims=True) / 7.0
    code_scales = np.maximum(code_scales, 1e-5)
    alpha = 0.5
    adjusted = (1.0 - alpha) * chat_scales + alpha * code_scales
    return adjusted
