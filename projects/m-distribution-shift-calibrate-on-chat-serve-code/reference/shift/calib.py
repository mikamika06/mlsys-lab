import numpy as np


def compute_chat_baseline(weights, chat_activations):
    scales = np.max(np.abs(chat_activations), axis=0, keepdims=True) / 7.0
    scales = np.maximum(scales, 1e-5)
    quantized = np.round(chat_activations / scales) * scales
    return scales


def measure_code_shift(weights, chat_scales, code_activations):
    approx = np.round(code_activations / chat_scales) * chat_scales
    err = np.linalg.norm(code_activations - approx) / (np.linalg.norm(code_activations) + 1e-8)
    return float(err)
