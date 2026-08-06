import numpy as np


def compute_scales(activations):
    return np.max(np.abs(activations), axis=(0, 1)) / 7.0
