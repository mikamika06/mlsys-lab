import math
from vllm_budget.kv import bytes_per_token


def max_context_length(
    config: dict,
    dtype: str,
    model_weight_bytes: int,
    non_model_overhead_bytes: int,
    total_vram_bytes: int,
) -> int:
    bpt = bytes_per_token(config, dtype)
    avail = total_vram_bytes - model_weight_bytes - non_model_overhead_bytes
    if avail <= 0 or bpt <= 0:
        return 0
    return avail // bpt
