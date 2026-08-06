import numpy as np


def quantize_tensor(W, num_bits):
    """Uniform symmetric/affine quantization of floating point matrix W."""
    raise NotImplementedError


def prune_tensor(W, sparsity):
    """Magnitude prune rows of matrix W to given ratio."""
    raise NotImplementedError


def compare_order_error(W, X, sparsity, num_bits):
    """Returns dictionary with keys 'ptq_mse' and 'qtp_mse' representing output reconstruction error."""
    raise NotImplementedError
