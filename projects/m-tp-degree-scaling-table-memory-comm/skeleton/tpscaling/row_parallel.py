def row_parallel_forward(x_splits, weight_splits, bias=None):
    raise NotImplementedError


def row_parallel_backward(grad_output_splits, x_splits, weight_splits):
    raise NotImplementedError
