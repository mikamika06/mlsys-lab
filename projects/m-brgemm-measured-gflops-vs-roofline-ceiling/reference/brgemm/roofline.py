def compute_memory_traffic(M, N, K, mb, nb, kb, bytes_per_elem=2):
    """Compute DRAM memory traffic in bytes considering BRGeMM tile reuse."""
    n_blocks = (N + nb - 1) // nb
    m_blocks = (M + mb - 1) // mb
    a_bytes = M * K * n_blocks * bytes_per_elem
    b_bytes = K * N * m_blocks * bytes_per_elem
    c_bytes = 2 * M * N * bytes_per_elem
    return a_bytes + b_bytes + c_bytes


def analyze_roofline(M, N, K, mb, nb, kb, peak_gflops, bw_gbps, runtime_ms, bytes_per_elem=2):
    """Analyze roofline model metrics and hardware efficiency for BRGeMM execution."""
    flops = 2 * M * N * K
    dram_bytes = compute_memory_traffic(M, N, K, mb, nb, kb, bytes_per_elem)
    ai = flops / dram_bytes
    ceiling = min(peak_gflops, ai * bw_gbps)
    measured = flops / (runtime_ms * 1e6)
    efficiency = measured / ceiling
    return {
        "flops": float(flops),
        "dram_bytes": float(dram_bytes),
        "arithmetic_intensity": float(ai),
        "roofline_ceiling_gflops": float(ceiling),
        "measured_gflops": float(measured),
        "efficiency": float(efficiency),
    }
