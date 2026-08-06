def tune_parameters(default_config: dict) -> dict:
    optimized = default_config.copy()
    optimized["threads"] = max(2, default_config.get("threads", 4) // 2)
    optimized["batch_size"] = default_config.get("batch_size", 512) * 2
    optimized["ubatch_size"] = 512
    optimized["pp_throughput"] = default_config.get("pp_throughput", 100.0) * 1.25
    return optimized
