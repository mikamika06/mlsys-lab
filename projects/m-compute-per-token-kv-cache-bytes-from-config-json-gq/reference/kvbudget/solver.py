from kvbudget.calculator import compute_per_token_kv_bytes


def solve_max_model_len(
    config: dict,
    vram_budget_bytes: int,
    max_num_seqs: int,
    kv_cache_dtype: str = "auto",
) -> int:
    """Compute the maximum integer max_model_len that fits in budget."""
    bytes_per_token = compute_per_token_kv_bytes(config, kv_cache_dtype)
    total_tokens = vram_budget_bytes // bytes_per_token
    return total_tokens // max_num_seqs


def predict_num_gpu_blocks(
    config: dict,
    total_gpu_memory_bytes: int,
    gpu_memory_utilization: float,
    model_weight_bytes: int,
    block_size: int = 16,
    kv_cache_dtype: str = "auto",
) -> int:
    """Predict vLLM num_gpu_blocks allocated at startup."""
    usable_memory = int(total_gpu_memory_bytes * gpu_memory_utilization)
    kv_memory_budget = usable_memory - model_weight_bytes
    if kv_memory_budget <= 0:
        return 0

    bytes_per_token = compute_per_token_kv_bytes(config, kv_cache_dtype)
    bytes_per_block = bytes_per_token * block_size
    return kv_memory_budget // bytes_per_block
