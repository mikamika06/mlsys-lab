def run_sweep(model_config, cpu_expert_counts, token_loads):
    results = []
    total_experts = model_config.get("total_experts", 8)
    base_memory = model_config.get("base_memory_bytes", 1024 * 1024 * 1024)
    expert_size = model_config.get("expert_size_bytes", 64 * 1024 * 1024)
    base_throughput = model_config.get("base_throughput", 100.0)

    for n_cpu in cpu_expert_counts:
        device_experts = max(0, total_experts - n_cpu)
        memory_usage = base_memory + (device_experts * expert_size)
        penalty = 1.0 - (n_cpu / (total_experts * 2.0))
        for load in token_loads:
            throughput = base_throughput * penalty * (load / (load + 10.0))
            results.append({
                "n_cpu": n_cpu,
                "token_load": load,
                "memory_usage": memory_usage,
                "throughput": throughput
            })
    return results
