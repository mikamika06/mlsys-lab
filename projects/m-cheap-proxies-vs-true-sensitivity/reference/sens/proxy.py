import numpy as np


def compute_proxy(layer):
    w = layer["weight"]
    return float(np.mean(np.abs(w)))
