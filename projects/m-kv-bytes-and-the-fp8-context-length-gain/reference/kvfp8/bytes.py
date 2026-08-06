import numpy as np


def compute_kv_cache_bytes(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int = 1,
    fp8: bool = False,
) -> int:
    element_size = 1 if fp8 else 2
    bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * element_size
    return int(batch_size * seq_len * bytes_per_token)


def max_context_length(
    gpu_memory_bytes: int,
    model_weight_bytes: int,
    activation_budget_bytes: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    batch_size: int = 1,
    fp8: bool = False,
) -> int:
    avail_bytes = gpu_memory_bytes - model_weight_bytes - activation_budget_bytes
    if avail_bytes <= 0:
        return 0
    element_size = 1 if fp8 else 2
    bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * element_size
    bytes_per_seq_pos = batch_size * bytes_per_token
    return int(avail_bytes // bytes_per_seq_pos)
