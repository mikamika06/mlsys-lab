import numpy as np


def custom_fused_op_impl(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    raise NotImplementedError


class CustomOpRegistry:
    def __init__(self):
        self.registered = False

    def register(self):
        raise NotImplementedError

    def meta_impl(self, x_shape, weight_shape):
        raise NotImplementedError
