def predict_checkpoint_size(model_config: dict, scheme: dict) -> int:
    hidden_size = model_config.get("hidden_size", 4096)
    num_hidden_layers = model_config.get("num_hidden_layers", 32)
    intermediate_size = model_config.get("intermediate_size", 11024)
    vocab_size = model_config.get("vocab_size", 32000)
    num_attention_heads = model_config.get("num_attention_heads", 32)
    num_key_value_heads = model_config.get("num_key_value_heads", 32)

    bits_w = scheme.get("bits_w", 16)
    bits_a = scheme.get("bits_a", 16)
    group_size = scheme.get("group_size", -1)

    attn_weight_params = num_hidden_layers * (
        num_attention_heads * (hidden_size * (hidden_size // num_attention_heads))
        + 2 * num_key_value_heads * (hidden_size * (hidden_size // num_attention_heads))
        + hidden_size * hidden_size
    )
    mlp_weight_params = num_hidden_layers * (
        2 * hidden_size * intermediate_size + intermediate_size * hidden_size
    )
    other_params = vocab_size * hidden_size + num_hidden_layers * hidden_size * 2

    total_weight_params = attn_weight_params + mlp_weight_params + other_params

    bytes_per_weight = bits_w / 8.0
    if bits_w < 16 and group_size > 0:
        scale_overhead_ratio = 2.0 / group_size
        bytes_per_weight += scale_overhead_ratio

    raw_weight_bytes = int(total_weight_params * bytes_per_weight)
    overhead_bytes = int(num_hidden_layers * 1024 * (bits_a / 16.0))
    return raw_weight_bytes + overhead_bytes
