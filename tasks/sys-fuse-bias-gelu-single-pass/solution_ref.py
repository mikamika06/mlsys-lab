import numpy as np

def fuse_bias_gelu(x, bias):
    y = x + bias
    sqrt2pi = np.sqrt(2/np.pi)
    return sqrt2pi * y * (1 + np.tanh(sqrt2pi*(y + 0.044715*y**3)))
