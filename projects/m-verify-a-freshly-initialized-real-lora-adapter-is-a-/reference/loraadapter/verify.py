import numpy as np


def verify_no_op(layer, x, tol=1e-7):
    base_out = x @ layer.weight.T
    out = layer.forward(x)
    max_err = float(np.max(np.abs(out - base_out)))
    return max_err <= tol, max_err
