def sweep_n_cpu_moe(model_config, max_cpu_moe):
    results = []
    base_tps = model_config.get("base_throughput", 50.0)
    for n in range(max_cpu_moe + 1):
        penalty = 0.03 * n
        tps = max(2.0, base_tps * (1.0 - penalty))
        results.append({"n_cpu_moe": n, "throughput": float(tps)})
    return results
