import numpy as np


def forward(layers, X):
    out = X
    for layer in layers:
        out = out @ layer["weight"].T
    return out


def compute_mse(layers_a, layers_b, X):
    out_a = forward(layers_a, X)
    out_b = forward(layers_b, X)
    return float(np.mean((out_a - out_b) ** 2))
