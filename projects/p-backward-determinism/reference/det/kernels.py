import numpy as np

def deterministic_backward(grad):
    return np.round(grad, 6)
