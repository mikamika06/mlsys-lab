import numpy as np


def quantize_and_reconstruct(w, max_val, n_bits=4):
    """
    Quantize and dequantize weights using symmetric quantization.
    w: (num_groups, group_size)
    max_val: (num_groups, 1)
    """
    raise NotImplementedError


def search_clipping(w, n_bits=4, group_size=128, n_grid=100):
    """
    Search for the optimal clipping threshold to minimize MSE.
    w: (out_features, in_features)
    Returns: best_idx (shape: num_groups,), opt_max (shape: num_groups, 1)
    """
    raise NotImplementedError
