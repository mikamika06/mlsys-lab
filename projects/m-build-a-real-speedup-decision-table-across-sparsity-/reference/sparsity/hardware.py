import numpy as np


def validate_nm_tensorcore_alignment(M, N, K, dtype_bits=16):
    k_alignment = 64 if dtype_bits == 8 else 32
    m_alignment = 16
    n_alignment = 16

    m_valid = (M % m_alignment == 0) and (M >= m_alignment)
    n_valid = (N % n_alignment == 0) and (N >= n_alignment)
    k_valid = (K % k_alignment == 0) and (K >= k_alignment)

    valid = m_valid and n_valid and k_valid
    return {
        "valid": valid,
        "m_valid": m_valid,
        "n_valid": n_valid,
        "k_valid": k_valid,
        "required_k_multiple": k_alignment,
    }


def compute_nm_speedup_gap(M, N, K, sparsity, bandwidth_gbps, compute_tflops, dtype_bits=16):
    align = validate_nm_tensorcore_alignment(M, N, K, dtype_bits)

    flops = 2.0 * M * N * K
    bytes_per_elem = dtype_bits / 8.0
    dense_bytes = (M * K + K * N + M * N) * bytes_per_elem

    dense_compute_time = flops / (compute_tflops * 1e12)
    dense_memory_time = dense_bytes / (bandwidth_gbps * 1e9)
    dense_time = max(dense_compute_time, dense_memory_time)

    sparse_weights_bytes = (M * K * 0.5 * bytes_per_elem) + (M * K * 0.5 * 2.0 / 8.0)
    sparse_bytes = sparse_weights_bytes + ((K * N + M * N) * bytes_per_elem)

    if align["valid"] and abs(sparsity - 0.5) < 1e-5:
        sparse_flops = flops * 0.5
        efficiency = 0.85
        sparse_compute_time = sparse_flops / (compute_tflops * 2.0 * 1e12 * efficiency)
        sparse_memory_time = sparse_bytes / (bandwidth_gbps * 1e9)
        sparse_time = max(sparse_compute_time, sparse_memory_time)
        achievable_speedup = dense_time / sparse_time
    else:
        achievable_speedup = 1.0

    theoretical_speedup = 2.0 if abs(sparsity - 0.5) < 1e-5 else (1.0 / max(1.0 - sparsity, 1e-5))

    return {
        "theoretical_speedup": float(theoretical_speedup),
        "achievable_speedup": float(achievable_speedup),
        "speedup_gap": float(theoretical_speedup - achievable_speedup),
        "is_aligned": align["valid"],
    }
