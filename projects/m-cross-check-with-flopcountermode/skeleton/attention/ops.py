import numpy as np
from attention.tracker import FlopCounterMode


def matmul(shape_a, shape_b):
    """Records FLOPs for A @ B and returns output shape."""
    raise NotImplementedError


def softmax(shape):
    """Records FLOPs for softmax over the last dimension."""
    raise NotImplementedError
