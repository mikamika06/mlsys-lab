import numpy as np

CONFIGS = [
    {
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "head_dim": 128,
        "num_layers": 32,
    },
    {
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "num_layers": 32,
    },
    {
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "num_layers": 40,
    },
    {
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 64,
        "num_layers": 12,
    },
]


def classify_attention(config: dict) -> dict:
    q_heads = config["num_attention_heads"]
    kv_heads = config.get("num_key_value_heads", q_heads)

    if kv_heads == q_heads:
        family = "MHA"
    elif kv_heads == 1:
        family = "MQA"
    elif 1 < kv_heads < q_heads:
        family = "GQA"
    else:
        raise ValueError("Invalid head configuration")

    ratio = q_heads // kv_heads
    return {
        "family": family,
        "num_q_heads": q_heads,
        "num_kv_heads": kv_heads,
        "group_ratio": ratio,
    }


def compute_group_mapping(config: dict) -> list[int]:
    q_heads = config["num_attention_heads"]
    kv_heads = config.get("num_key_value_heads", q_heads)
    ratio = q_heads // kv_heads
    return [q_idx // ratio for q_idx in range(q_heads)]


def calculate_kv_cache_bytes(
    config: dict, batch_size: int, seq_len: int, dtype_bytes: int = 2
) -> dict:
    layers = config.get("num_hidden_layers", config.get("num_layers", 1))
    q_heads = config["num_attention_heads"]
    kv_heads = config.get("num_key_value_heads", q_heads)
    head_dim = config["head_dim"]

    elements_per_head = batch_size * seq_len * head_dim * 2 * layers
    mha_equivalent_bytes = elements_per_head * q_heads * dtype_bytes
    native_gqa_bytes = elements_per_head * kv_heads * dtype_bytes
    bytes_saved = mha_equivalent_bytes - native_gqa_bytes

    return {
        "mha_bytes": mha_equivalent_bytes,
        "native_bytes": native_gqa_bytes,
        "bytes_saved": bytes_saved,
        "savings_ratio": (
            bytes_saved / mha_equivalent_bytes if mha_equivalent_bytes else 0.0
        ),
    }


def analyze_gpu_expansion_overhead(
    config: dict, batch_size: int, seq_len: int, dtype_bytes: int = 2
) -> dict:
    kv_info = calculate_kv_cache_bytes(config, batch_size, seq_len, dtype_bytes)
    native_bytes = kv_info["native_bytes"]
    mha_bytes = kv_info["mha_bytes"]

    overhead_bytes = mha_bytes - native_bytes
    expansion_factor = mha_bytes / native_bytes if native_bytes > 0 else 1.0

    return {
        "native_storage_bytes": native_bytes,
        "expanded_runtime_bytes": mha_bytes,
        "expansion_overhead_bytes": overhead_bytes,
        "expansion_factor": expansion_factor,
    }
