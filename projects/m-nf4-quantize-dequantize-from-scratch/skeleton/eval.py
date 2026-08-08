import numpy as np
from nf4 import build_nf4_codebook, quantize_tensor, dequantize_tensor


def compare_distributions(w_normal, w_uniform, w_laplace, block_size):
    """
    Evaluate Mean Squared Error of NF4, INT4, and FP4 codebooks.

    INT4 codebook to test: np.linspace(-1, 1, 16)
    FP4 codebook to test: np.array([-6, -4, -3, -2, -1.5, -1, -0.5, 0.0, 0.0, 0.5, 1, 1.5, 2, 3, 4, 6]) / 6.0

    Returns a dict mapping the distribution name to a dict of codebook MSEs.
    e.g. {"normal": {"nf4": 0.01, "int4": 0.02, "fp4": 0.03}, ...}
    """
    raise NotImplementedError
