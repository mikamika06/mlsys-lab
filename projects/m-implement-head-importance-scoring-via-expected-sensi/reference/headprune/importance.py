import numpy as np

def compute_importance(activations, gradients):
    return np.mean(np.abs(activations * gradients), axis=(0, 2))
