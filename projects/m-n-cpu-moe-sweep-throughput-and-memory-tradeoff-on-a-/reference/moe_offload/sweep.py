def run_sweep(model_cfg, n_cpu_list):
    results = []
    base_t = model_cfg["base_throughput"]
    vram = model_cfg["vram_limit_gb"]
    tot = model_cfg["total_experts"]
    for n in n_cpu_list:
        fraction_gpu = (tot - n) / tot
        vram_used = vram * (0.3 + 0.7 * fraction_gpu)
        throughput = base_t * (0.4 + 0.6 * fraction_gpu)
        results.append({"n_cpu_moe": n, "throughput": float(throughput), "vram_gb": float(vram_used)})
    return results
