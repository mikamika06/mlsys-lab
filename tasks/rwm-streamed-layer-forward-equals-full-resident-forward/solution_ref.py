import numpy as np


def streamed_mlp_forward(layers, x):
    out = np.asarray(x, dtype=np.float64)
    for i, layer in enumerate(layers):
        w = layer["w"]
        b = layer["b"]
        out = out @ w + b
        del w
        del b
        if i != len(layers) - 1:
            out = np.maximum(out, 0)
    return out
