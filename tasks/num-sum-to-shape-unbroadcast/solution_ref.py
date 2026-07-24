import numpy as np

def sum_to_shape(grad, input_shape):
    grad = np.asarray(grad, dtype=np.float64)
    target_shape = grad.shape
    ndim_diff = len(target_shape) - len(input_shape)
    padded = (1,) * ndim_diff + tuple(input_shape)
    axes_to_sum = tuple(
        i for i in range(len(target_shape))
        if padded[i] == 1 and target_shape[i] > 1
    )
    if axes_to_sum:
        grad = grad.sum(axis=axes_to_sum, keepdims=True)
    return grad.reshape(input_shape)
