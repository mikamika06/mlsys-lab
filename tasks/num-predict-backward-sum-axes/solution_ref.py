import numpy as np

def predict_backward_sum_axes(a_shape, b_shape):
    a = np.ones(a_shape)
    b = np.ones(b_shape)
    try:
        c = a * b
    except ValueError:
        return ((0,), (0,))  # or ((0,), (0,))
    
    a_axes = tuple(i for i, s in enumerate(a_shape) if s != 1 and c.shape[i] != s)
    b_axes = tuple(i for i, s in enumerate(b_shape) if s != 1 and c.shape[i] != s)
    return (a_axes, b_axes)
