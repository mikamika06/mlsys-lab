CONFIGS = [
    {"M": 256, "N": 256, "K": 256, "mb": 32, "nb": 32, "kb": 32, "peak": 500.0, "bw": 50.0, "runtime": 1.2},
    {"M": 512, "N": 1024, "K": 512, "mb": 64, "nb": 64, "kb": 64, "peak": 1200.0, "bw": 80.0, "runtime": 2.5},
    {"M": 1000, "N": 1000, "K": 1000, "mb": 64, "nb": 64, "kb": 32, "peak": 800.0, "bw": 40.0, "runtime": 6.0},
    {"M": 128, "N": 512, "K": 2048, "mb": 32, "nb": 64, "kb": 128, "peak": 1500.0, "bw": 120.0, "runtime": 1.0},
]


def reconstruct_call_sequence(M, N, K, mb, nb, kb):
    calls = []
    for m in range(0, M, mb):
        m_len = min(mb, M - m)
        for n in range(0, N, nb):
            n_len = min(nb, N - n)
            batch_size = (K + kb - 1) // kb
            a_offsets = []
            b_offsets = []
            for k_idx in range(batch_size):
                k_start = k_idx * kb
                a_off = m * K + k_start
                b_off = k_start * N + n
                a_offsets.append(a_off)
                b_offsets.append(b_off)
            c_off = m * N + n
            calls.append({
                "m_start": m,
                "n_start": n,
                "m_len": m_len,
                "n_len": n_len,
                "batch_size": batch_size,
                "a_offsets": a_offsets,
                "b_offsets": b_offsets,
                "c_offset": c_off,
            })
    return calls


def compute_memory_traffic(M, N, K, mb, nb, kb, bytes_per_elem=2):
    n_blocks = (N + nb - 1) // nb
    m_blocks = (M + mb - 1) // mb
    a_bytes = M * K * n_blocks * bytes_per_elem
    b_bytes = K * N * m_blocks * bytes_per_elem
    c_bytes = 2 * M * N * bytes_per_elem
    return a_bytes + b_bytes + c_bytes


def analyze_roofline(M, N, K, mb, nb, kb, peak_gflops, bw_gbps, runtime_ms, bytes_per_elem=2):
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
