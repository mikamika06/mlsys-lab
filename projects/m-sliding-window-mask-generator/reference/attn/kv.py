from typing import Any, Dict, List


def calculate_kv_bytes(
    config: Dict[str, Any],
    seq_len: int,
    batch_size: int = 1,
    dtype_bytes: int = 2,
) -> Dict[str, int]:
    num_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    element_size = 2 * num_heads * head_dim * dtype_bytes * batch_size

    total_bytes = 0
    per_layer = {}

    for i, layer in enumerate(config["layers"]):
        kind = layer.get("kind", "full")
        if kind == "sliding":
            w = layer["window_size"]
            s = layer.get("num_sinks", 0)
            effective_tokens = min(seq_len, w + s)
        else:
            effective_tokens = seq_len

        layer_bytes = effective_tokens * element_size
        per_layer[f"layer_{i}"] = layer_bytes
        total_bytes += layer_bytes

    return {"total_bytes": total_bytes, "per_layer": per_layer}


def calculate_memory_savings(
    config: Dict[str, Any],
    max_seq_len: int,
    batch_size: int = 1,
    dtype_bytes: int = 2,
) -> float:
    actual = calculate_kv_bytes(config, max_seq_len, batch_size, dtype_bytes)["total_bytes"]
    num_layers = len(config["layers"])
    num_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    full_bytes = num_layers * max_seq_len * 2 * num_heads * head_dim * dtype_bytes * batch_size

    if full_bytes == 0:
        return 0.0
    return float((full_bytes - actual) / full_bytes)
