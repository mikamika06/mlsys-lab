import numpy as np


def quantize_per_tensor(weights, qmin=-128, qmax=127):
    raise NotImplementedError


def quantize_per_channel(weights, qmin=-128, qmax=127, axis=0):
    raise NotImplementedError


def measure_quantization_error(weights, dequantized_weights):
    raise NotImplementedError
