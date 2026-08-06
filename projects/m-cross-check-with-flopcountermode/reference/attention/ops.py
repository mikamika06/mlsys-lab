import numpy as np
from attention.tracker import FlopCounterMode


def matmul(shape_a, shape_b):
    """Records FLOPs for A @ B and returns output shape."""
    batch_a = shape_a[:-2]
    batch_b = shape_b[:-2]
    batch_out = np.broadcast_shapes(batch_a, batch_b)

    m, k = shape_a[-2:]
    k2, n = shape_b[-2:]
    if k != k2:
        raise ValueError("Inner dims must match")

    flops = 2 * int(np.prod(batch_out)) * m * k * n
    FlopCounterMode.record(flops)

    return tuple(batch_out) + (m, n)


def softmax(shape):
    """Records FLOPs for softmax over the last dimension."""
    flops = 5 * int(np.prod(shape))
    FlopCounterMode.record(flops)
    return shape
