import numpy as np


class QLoRALinear:
    def __init__(self, in_features, out_features, r=4, alpha=8.0, block_size=16):
        raise NotImplementedError

    def quantize_base(self, weight):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def backward(self, x, grad_output):
        raise NotImplementedError


def apply_qlora(model_dict, target_keys, r=4, alpha=8.0, block_size=16):
    raise NotImplementedError
