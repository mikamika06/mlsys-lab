import numpy as np


def compute_rel_err(got, want):
    """
    Compute max relative error between arrays got and want.
    """
    raise NotImplementedError


def verify_tolerance_bounds(got, want, rtol=1e-5, atol=1e-8):
    """
    Verify relative and absolute error boundaries.
    """
    raise NotImplementedError


def analyze_error_vs_seqlen(query_dim, seqlens, chunk_size=64, seed=42):
    """
    Measure max relative error across sequence lengths.
    """
    raise NotImplementedError
