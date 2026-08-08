def solve_max_model_len(
    config: dict,
    vram_budget_bytes: int,
    max_num_seqs: int,
    kv_cache_dtype: str = "auto",
) -> int:
    """Compute the maximum integer max_model_len that fits in budget."""
    raise NotImplementedError


def predict_num_gpu_blocks(
    config: dict,
    total_gpu_memory_bytes: int,
    gpu_memory_utilization: float,
    model_weight_bytes: int,
    block_size: int = 16,
    kv_cache_dtype: str = "auto",
) -> int:
    """Predict vLLM num_gpu_blocks allocated at startup."""
    raise NotImplementedError
