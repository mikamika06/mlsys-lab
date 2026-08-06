import numpy as np


def compute_theoretical_bytes(shape, dtype_bits, format_type, sparsity):
    """
    Calculates theoretical storage in bytes for dense, CSR, COO, and 2:4 formats.
    """
    raise NotImplementedError


def dense_pt_saved_bytes(shape, dtype_bits, sparsity):
    """
    Returns the storage size in bytes when saving a dense PyTorch-style zeroed tensor.
    """
    raise NotImplementedError


def compute_csr_breakeven_sparsity(shape, dtype_bits, index_bits=32):
    """
    Computes the minimum sparsity ratio required for CSR to be smaller than Dense.
    """
    raise NotImplementedError
