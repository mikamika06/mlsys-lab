import numpy as np


def decide_quantization_strategy(
    num_weights: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int,
    max_memory_bytes: float,
    weight_bit_options: list[int],
    kv_bit_options: list[int],
    weight_mse_fn,
    kv_mse_fn,
) -> dict:
    raise NotImplementedError
