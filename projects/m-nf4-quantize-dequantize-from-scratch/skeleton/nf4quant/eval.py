"""Quantization error evaluation across distributions."""

import numpy as np


def quantization_error(x, block_size, codebook):
    """Compute MSE reconstruction error for blockwise quantization."""
    raise NotImplementedError


def compare_codebooks_on_distributions(distributions, block_size):
    """Compare NF4, FP4, and INT4 quantization error on multiple distributions."""
    raise NotImplementedError
