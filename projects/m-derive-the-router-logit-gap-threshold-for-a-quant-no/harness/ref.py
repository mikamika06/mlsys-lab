import numpy as np


def generate_inputs():
    np.random.seed(42)
    weights = np.random.randn(16, 8)
    quantized_weights = weights + np.random.uniform(-0.05, 0.05, size=weights.shape)
    hidden_states = np.random.randn(4, 16)
    logits = np.dot(hidden_states, weights)
    return weights, quantized_weights, hidden_states, logits
