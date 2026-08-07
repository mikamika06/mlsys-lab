import numpy as np
from fp8kv.quant import simulate_e4m3, get_per_tensor_scale, get_per_head_scale


def measure_rel_err(orig, approx):
    """
    Compute mean(abs(orig - approx)) / mean(abs(orig)).
    If the denominator is exactly zero, return 0.0.
    """
    raise NotImplementedError


def find_breaking_head(x, max_val=448.0):
    """
    Compare per-tensor vs per-head quantization relative error for each head.
    Return the integer index of the head that suffers the LARGEST increase
    in relative error when switching from per-head to per-tensor scaling.
    """
    raise NotImplementedError
