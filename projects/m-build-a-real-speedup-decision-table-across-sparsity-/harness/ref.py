import numpy as np

CONFIGS = [
    {"shape": (1024, 1024), "dtype_bits": 16, "sparsity": 0.5},
    {"shape": (2048, 4096), "dtype_bits": 16, "sparsity": 0.85},
    {"shape": (512, 512), "dtype_bits": 8, "sparsity": 0.5},
    {"shape": (4096, 4096), "dtype_bits": 16, "sparsity": 0.92},
]

HW_CASES = [
    {"M": 128, "N": 128, "K": 64, "sparsity": 0.5, "dtype_bits": 16, "bandwidth_gbps": 900.0, "compute_tflops": 312.0},
    {"M": 128, "N": 128, "K": 20, "sparsity": 0.5, "dtype_bits": 16, "bandwidth_gbps": 900.0, "compute_tflops": 312.0},
    {"M": 16, "N": 16, "K": 64, "sparsity": 0.5, "dtype_bits": 8, "bandwidth_gbps": 1500.0, "compute_tflops": 640.0},
    {"M": 256, "N": 256, "K": 128, "sparsity": 0.8, "dtype_bits": 16, "bandwidth_gbps": 900.0, "compute_tflops": 312.0},
]


def ref_compute_theoretical_bytes(shape, dtype_bits, format_type, sparsity):
    M, N = shape
    total = M * N
    nnz = int(round(total * (1.0 - sparsity)))
    val_b = (nnz * dtype_bits) / 8.0

    if format_type == "dense":
        return float((total * dtype_bits) // 8)
    elif format_type == "csr":
        return float(val_b + ((M + 1) * 32) / 8.0 + (nnz * 32) / 8.0)
    elif format_type == "coo":
        return float(val_b + (2 * nnz * 32) / 8.0)
    elif format_type == "2:4":
        return float((total * 0.5 * dtype_bits) / 8.0 + (total * 0.5 * 2) / 8.0)


def ref_dense_pt_saved_bytes(shape, dtype_bits, sparsity):
    return float((shape[0] * shape[1] * dtype_bits) // 8)


def ref_compute_csr_breakeven_sparsity(shape, dtype_bits, index_bits=32):
    M, N = shape
    total = M * N
    dense_b = (total * dtype_bits) / 8.0
    row_ptrs_b = ((M + 1) * index_bits) / 8.0
    per_nnz_b = (dtype_bits + index_bits) / 8.0
    max_nnz = (dense_b - row_ptrs_b) / per_nnz_b
    if max_nnz <= 0:
        return 1.0
    return float(max(0.0, min(1.0, 1.0 - (max_nnz / total))))


def ref_validate_nm_alignment(M, N, K, dtype_bits=16):
    k_mult = 64 if dtype_bits == 8 else 32
    m_v = (M % 16 == 0) and (M >= 16)
    n_v = (N % 16 == 0) and (N >= 16)
    k_v = (K % k_mult == 0) and (K >= k_mult)
    return {
        "valid": bool(m_v and n_v and k_v),
        "m_valid": bool(m_v),
        "n_valid": bool(n_v),
        "k_valid": bool(k_v),
        "required_k_multiple": k_mult,
    }


def ref_compute_nm_speedup_gap(M, N, K, sparsity, bandwidth_gbps, compute_tflops, dtype_bits=16):
    align = ref_validate_nm_alignment(M, N, K, dtype_bits)
    flops = 2.0 * M * N * K
    b_elem = dtype_bits / 8.0
    dense_b = (M * K + K * N + M * N) * b_elem

    dense_comp = flops / (compute_tflops * 1e12)
    dense_mem = dense_b / (bandwidth_gbps * 1e9)
    dense_t = max(dense_comp, dense_mem)

    sparse_w_b = (M * K * 0.5 * b_elem) + (M * K * 0.5 * 2.0 / 8.0)
    sparse_b = sparse_w_b + ((K * N + M * N) * b_elem)

    if align["valid"] and abs(sparsity - 0.5) < 1e-5:
        sparse_flops = flops * 0.5
        sparse_comp = sparse_flops / (compute_tflops * 2.0 * 1e12 * 0.85)
        sparse_mem = sparse_b / (bandwidth_gbps * 1e9)
        sparse_t = max(sparse_comp, sparse_mem)
        achievable = dense_t / sparse_t
    else:
        achievable = 1.0

    theo = 2.0 if abs(sparsity - 0.5) < 1e-5 else (1.0 / max(1.0 - sparsity, 1e-5))

    return {
        "theoretical_speedup": float(theo),
        "achievable_speedup": float(achievable),
        "speedup_gap": float(theo - achievable),
        "is_aligned": align["valid"],
    }
