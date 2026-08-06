def predict_num_gpu_blocks(
    config: dict,
    dtype: str,
    total_vram_bytes: int,
    gpu_memory_utilization: float,
    model_weight_bytes: int,
    non_model_overhead_bytes: int,
    block_size: int,
) -> int:
    """Predicts vLLM num_gpu_blocks for given memory and block params."""
    raise NotImplementedError
