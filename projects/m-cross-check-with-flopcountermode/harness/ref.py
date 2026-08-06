import numpy as np


def ref_matmul_flops(shape_a, shape_b):
    b = np.broadcast_shapes(shape_a[:-2], shape_b[:-2])
    return 2 * int(np.prod(b)) * shape_a[-2] * shape_a[-1] * shape_b[-1]


def ref_softmax_flops(shape):
    return 5 * int(np.prod(shape))


def ref_analytical(b, h, seq, d):
    return 4 * b * h * seq * seq * d + 5 * b * h * seq * seq
