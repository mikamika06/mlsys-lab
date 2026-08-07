import numpy as np

CONFIGS = [
    {
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 11008,
        "hidden_size": 4096,
        "num_layers": 32,
        "dtype_bytes": 2
    },
    {
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "intermediate_size": 14336,
        "hidden_size": 4096,
        "num_layers": 30,
        "dtype_bytes": 2
    },
    {
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "intermediate_size": 28672,
        "hidden_size": 8192,
        "num_layers": 80,
        "dtype_bytes": 2
    },
    {
        "num_attention_heads": 96,
        "num_key_value_heads": 16,
        "intermediate_size": 38400,
        "hidden_size": 12288,
        "num_layers": 96,
        "dtype_bytes": 2
    }
]

TP_DEGREES = [1, 2, 4, 8, 16]


def validate_tp_feasibility(config: dict, tp_size: int) -> dict:
    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    intermediate_size = config["intermediate_size"]

    heads_ok = (num_heads % tp_size == 0)
    kv_heads_ok = (num_kv_heads % tp_size == 0)
    intermediate_ok = (intermediate_size % tp_size == 0)

    is_feasible = heads_ok and kv_heads_ok and intermediate_ok
    reasons = []
    if not heads_ok:
        reasons.append(f"num_attention_heads ({num_heads}) not divisible by tp_size ({tp_size})")
    if not kv_heads_ok:
        reasons.append(f"num_key_value_heads ({num_kv_heads}) not divisible by tp_size ({tp_size})")
    if not intermediate_ok:
        reasons.append(f"intermediate_size ({intermediate_size}) not divisible by tp_size ({tp_size})")

    return {
        "is_feasible": is_feasible,
        "reasons": reasons
    }


def compute_tp_traffic(config: dict, tp_size: int, target_tokens_per_sec: float) -> dict:
    if tp_size <= 1:
        return {
            "bytes_per_token_per_rank": 0.0,
            "total_bus_bytes_per_sec": 0.0
        }

    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]
    dtype_bytes = config["dtype_bytes"]

    bytes_per_token_per_rank = 2 * (tp_size - 1) / tp_size * (2 * hidden_size * num_layers * dtype_bytes)
    total_bus_bytes_per_sec = bytes_per_token_per_rank * target_tokens_per_sec * tp_size

    return {
        "bytes_per_token_per_rank": float(bytes_per_token_per_rank),
        "total_bus_bytes_per_sec": float(total_bus_bytes_per_sec)
    }


def compute_pp_bubble_fraction(num_microbatches: int, num_pipeline_stages: int) -> float:
    if num_microbatches <= 0 or num_pipeline_stages <= 0:
        raise ValueError("Microbatches and pipeline stages must be positive integers.")
    if num_microbatches < num_pipeline_stages:
        raise ValueError("Microbatches must be greater than or equal to pipeline stages.")

    p = num_pipeline_stages
    m = num_microbatches
    return float((p - 1) / (m + p - 1))
