import numpy as np


def custom_fused_op_impl(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    res = np.dot(x, weight)
    res = np.maximum(res, 0.0)
    return res


class CustomOpRegistry:
    def __init__(self):
        self.registered = False

    def register(self):
        self.registered = True

    def meta_impl(self, x_shape, weight_shape):
        batch, seq_len, in_dim = x_shape
        w_in, out_dim = weight_shape
        if in_dim != w_in:
            raise ValueError("Dimension mismatch")
        return (batch, seq_len, out_dim)
