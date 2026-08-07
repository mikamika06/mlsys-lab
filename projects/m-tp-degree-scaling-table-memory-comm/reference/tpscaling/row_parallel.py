import numpy as np


def row_parallel_forward(x_splits, weight_splits, bias=None):
    tp_degree = len(x_splits)
    local_outputs = []
    for x, w in zip(x_splits, weight_splits):
        y_local = np.matmul(x, w)
        local_outputs.append(y_local)

    y_sum = np.sum(local_outputs, axis=0)
    if bias is not None:
        y_sum = y_sum + bias

    return [y_sum.copy() for _ in range(tp_degree)]


def row_parallel_backward(grad_output_splits, x_splits, weight_splits):
    tp_degree = len(x_splits)
    grad_x_splits = []
    grad_w_splits = []

    for i in range(tp_degree):
        g_out = grad_output_splits[i]
        x_i = x_splits[i]
        w_i = weight_splits[i]

        g_x = np.matmul(g_out, w_i.T)
        grad_x_splits.append(g_x)

        x_flat = x_i.reshape(-1, x_i.shape[-1])
        g_flat = g_out.reshape(-1, g_out.shape[-1])
        g_w = np.matmul(x_flat.T, g_flat)
        grad_w_splits.append(g_w)

    g_bias = grad_output_splits[0].sum(axis=(0, 1)) if grad_output_splits[0].ndim == 3 else grad_output_splits[0].sum(axis=0)

    return grad_x_splits, grad_w_splits, g_bias
