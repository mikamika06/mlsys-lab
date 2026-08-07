def zero3_memory_math(layers: list[int], num_gpus: int) -> dict:
    total = sum(layers)
    return {
        "sharded_bytes": 16 * total // num_gpus,
        "comm_per_gpu_bytes": 6 * total * (num_gpus - 1) // num_gpus,
        "baseline_peak_active_bytes": 2 * max(layers)
    }
