def find_optimal_ngl(layers, vram_limit_mb, layer_memory_mb):
    best_ngl = 0
    max_tps = 0.0
    for ngl in range(layers + 1):
        mem = ngl * layer_memory_mb
        if mem > vram_limit_mb:
            break
        cpu_layers = layers - ngl
        tps = (ngl * 15.0) + (cpu_layers * 2.0)
        if tps > max_tps:
            max_tps = tps
            best_ngl = ngl
    return best_ngl, max_tps


def calculate_throughput_ratio(offload_fraction, base_cpu_tps, base_gpu_tps, memory_cost, vram_limit):
    if memory_cost > vram_limit:
        return 0.0
    effective_tps = (1.0 - offload_fraction) * base_cpu_tps + offload_fraction * base_gpu_tps
    baseline = base_cpu_tps if base_cpu_tps > 0 else 1.0
    return effective_tps / baseline
