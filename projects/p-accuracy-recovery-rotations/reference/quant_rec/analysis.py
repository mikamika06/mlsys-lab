import numpy as np

def measure_layer_error(weights, quantized_weights):
    diff = weights - quantized_weights
    return float(np.mean(diff ** 2))
