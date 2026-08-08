import numpy as np
from .codebooks import build_int4_codebook, build_fp4_codebook, build_nf4_codebook
from .quantize import quantize_blockwise, dequantize_blockwise


def compute_error(tensor, codebook, block_size=64):
    """
    Quantize and dequantize the tensor, returning the mean squared error.
    """
    raise NotImplementedError()


def compare_distributions():
    """
    Generate three tensors of length 1024 (seed=42):
    - 'normal': np.random.randn(1024)
    - 'uniform': np.random.uniform(-1, 1, 1024)
    - 'laplace': np.random.laplace(0, 1, 1024)

    For each, compute the mean squared error with INT4, FP4, and NF4 codebooks.
    Return a nested dictionary mapping distribution name to codebook name to error:
    {
       'normal': {'int4': err, 'fp4': err, 'nf4': err},
       ...
    }
    """
    raise NotImplementedError()
