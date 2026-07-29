import numpy as np


def softcap_forward(x, cap):
    x = np.asarray(x, dtype=np.float64)
    return cap * np.tanh(x / cap)


def softcap_backward(grad_output, x, cap):
    x = np.asarray(x, dtype=np.float64)
    grad_output = np.asarray(grad_output, dtype=np.float64)
    t = np.tanh(x / cap)
    return grad_output * (1.0 - t * t)
