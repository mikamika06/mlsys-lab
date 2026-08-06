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
    reduced_shape = tuple(
        1 if i in axes_to_sum else target_shape[i]
        for i in range(len(target_shape))
    )
    if axes_to_sum:
        out = np.zeros(reduced_shape, dtype=np.float64)
        sum_shapes = tuple(target_shape[i] for i in axes_to_sum)
        for out_idx in np.ndindex(reduced_shape):
            acc = 0.0
            for sub_idx in np.ndindex(sum_shapes):
                target_idx = list(out_idx)
                for ax, s_idx in zip(axes_to_sum, sub_idx):
                    target_idx[ax] = s_idx
                acc += grad[tuple(target_idx)]
            out[out_idx] = acc
    else:
        out = grad
    return out.reshape(input_shape)
