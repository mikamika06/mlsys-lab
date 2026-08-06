import numpy as np


def compute_kv_cache_bytes(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int = 1,
    fp8: bool = False,
) -> int:
    raise NotImplementedError


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
    raise NotImplementedError
