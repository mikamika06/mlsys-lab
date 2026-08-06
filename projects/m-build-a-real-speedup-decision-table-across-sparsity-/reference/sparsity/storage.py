import numpy as np


def compute_theoretical_bytes(shape, dtype_bits, format_type, sparsity):
    M, N = shape
    total_elements = M * N
    non_zeros = int(round(total_elements * (1.0 - sparsity)))
    val_bytes = (non_zeros * dtype_bits) / 8.0

    if format_type == "dense":
        return float((total_elements * dtype_bits) // 8)
    elif format_type == "csr":
        row_ptrs_bytes = ((M + 1) * 32) / 8.0
        col_indices_bytes = (non_zeros * 32) / 8.0
        return float(val_bytes + row_ptrs_bytes + col_indices_bytes)
    elif format_type == "coo":
        coords_bytes = (2 * non_zeros * 32) / 8.0
        return float(val_bytes + coords_bytes)
    elif format_type == "2:4":
        val_24_bytes = (total_elements * 0.5 * dtype_bits) / 8.0
        meta_bytes = (total_elements * 0.5 * 2) / 8.0
        return float(val_24_bytes + meta_bytes)
    else:
        raise ValueError(f"Unknown format: {format_type}")


def dense_pt_saved_bytes(shape, dtype_bits, sparsity):
    M, N = shape
    return float((M * N * dtype_bits) // 8)


def compute_csr_breakeven_sparsity(shape, dtype_bits, index_bits=32):
    M, N = shape
    total_elements = M * N
    dense_bytes = (total_elements * dtype_bits) / 8.0
    row_ptrs_bytes = ((M + 1) * index_bits) / 8.0
    per_nnz_bytes = (dtype_bits + index_bits) / 8.0

    max_nnz = (dense_bytes - row_ptrs_bytes) / per_nnz_bytes
    if max_nnz <= 0:
        return 1.0
    min_sparsity = 1.0 - (max_nnz / total_elements)
    return float(max(0.0, min(1.0, min_sparsity)))
