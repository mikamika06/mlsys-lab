"""Decode memory traffic computation."""


def _compute_weight_bytes(config: dict, dtype_bytes: int = 2) -> float:
    h = config["hidden_size"]
    n_layers = config["num_hidden_layers"]
    n_heads = config["num_attention_heads"]
    n_kv_heads = config.get("num_key_value_heads", n_heads)
    d_head = h // n_heads
    i = config.get("intermediate_size", 4 * h)

    attn_params = h * (n_heads * d_head) + 2 * h * (n_kv_heads * d_head) + (n_heads * d_head) * h
    mlp_params = 3 * h * i

    total_params = n_layers * (attn_params + mlp_params)
    return float(total_params * dtype_bytes)


def compute_decode_bytes_per_step(config: dict, batch_size: int, context_len: int, dtype_bytes: int = 2) -> float:
    """Compute HBM memory bytes transferred per decode step."""
    weight_bytes = _compute_weight_bytes(config, dtype_bytes)

    n_layers = config["num_hidden_layers"]
    n_heads = config["num_attention_heads"]
    n_kv_heads = config.get("num_key_value_heads", n_heads)
    d_head = h_dim = config["hidden_size"] // n_heads

    kv_cache_elements_per_token = 2 * n_layers * n_kv_heads * d_head
    kv_read_bytes = batch_size * context_len * kv_cache_elements_per_token * dtype_bytes

    return float(weight_bytes + kv_read_bytes)
