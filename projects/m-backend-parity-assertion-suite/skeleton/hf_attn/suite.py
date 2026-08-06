import numpy as np


def build_repro_case(batch_size, seq_len, valid_lengths):
    """
    Initialize np.random.seed(42) and generate q, k, v as float32 random normal 
    arrays of shape (batch_size, seq_len, 64). mask is zeros with 1.0 up to valid_lengths.
    """
    raise NotImplementedError


def assert_parity_on_valid(q, k, v, mask, ref_fn, test_fn):
    """
    Computes max absolute diff between ref_fn and test_fn
    only on valid tokens. Returns float.
    """
    raise NotImplementedError
