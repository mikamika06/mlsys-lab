import numpy as np
from kvquant.alloc import allocate_bits_jointly, compute_total_memory_bytes


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
    best = allocate_bits_jointly(
        num_weights,
        num_layers,
        num_heads,
        head_dim,
        seq_len,
        batch_size,
        max_memory_bytes,
        weight_bit_options,
        kv_bit_options,
        weight_mse_fn,
        kv_mse_fn,
    )

    if best is None:
        return {"should_quantize_kv": False, "best_config": None}

    unquant_kv_bits = max(kv_bit_options)
    quant_kv_bits = best["kv_bits"]

    should_quantize = quant_kv_bits < unquant_kv_bits

    return {
        "should_quantize_kv": should_quantize,
        "best_config": best,
    }
