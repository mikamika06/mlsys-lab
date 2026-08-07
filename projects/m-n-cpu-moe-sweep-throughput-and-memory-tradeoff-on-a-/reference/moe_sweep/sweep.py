def simulate_sweep(model_cfg, n_cpu_values):
    results = []
    for n_cpu in n_cpu_values:
        vram_bytes = max(0, model_cfg["total_vram"] - n_cpu * model_cfg["expert_bytes"])
        throughput = model_cfg["base_throughput"] * (1.0 - 0.02 * (n_cpu / model_cfg["total_experts"]))
        results.append({"n_cpu": n_cpu, "vram_bytes": vram_bytes, "throughput": throughput})
    return results
