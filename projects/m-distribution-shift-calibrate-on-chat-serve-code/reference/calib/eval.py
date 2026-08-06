import numpy as np


def evaluate_quantization(scales, code_activations):
    losses = []
    for scale, act in zip(scales, code_activations):
        quantized = np.round(act / scale) * scale
        mse = float(np.mean((act - quantized) ** 2))
        losses.append(mse)
    return float(np.mean(losses))
