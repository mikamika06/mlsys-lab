import numpy as np


def validate_nm_tensorcore_alignment(M, N, K, dtype_bits=16):
    """
    Validates hardware alignment constraints for 2:4 sparse Tensor Core GEMM.
    """
    raise NotImplementedError


def compute_nm_speedup_gap(M, N, K, sparsity, bandwidth_gbps, compute_tflops, dtype_bits=16):
    """
    Computes theoretical vs achievable speedup for 2:4 structured sparsity.
    """
    raise NotImplementedError
