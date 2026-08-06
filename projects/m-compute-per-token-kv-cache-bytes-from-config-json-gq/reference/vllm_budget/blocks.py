import math
from vllm_budget.kv import bytes_per_token


def predict_num_gpu_blocks(
    config: dict,
    dtype: str,
    total_vram_bytes: int,
    gpu_memory_utilization: float,
    model_weight_bytes: int,
    non_model_overhead_bytes: int,
    block_size: int,
) -> int:
    bpt = bytes_per_token(config, dtype)
    block_bytes = bpt * block_size
    usable_vram = int(math.floor(total_vram_bytes * gpu_memory_utilization))
    kv_budget = usable_vram - model_weight_bytes - non_model_overhead_bytes
    if kv_budget <= 0 or block_bytes <= 0:
        return 0
    return kv_budget // block_bytes
