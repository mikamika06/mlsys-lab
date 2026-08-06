def max_capacity_fp8(total_vram_bytes: int, model_weight_bytes: int, overhead_bytes: int, block_size: int, block_bytes_fp16: int, block_bytes_fp8: int, gpu_memory_utilization: float) -> int:
    available = int(total_vram_bytes * gpu_memory_utilization)
    usable = available - model_weight_bytes - overhead_bytes
    if usable <= 0:
        return 0
    return usable // block_bytes_fp8
