def compute_decode_bytes(config: dict, batch_size: int, context_len: int) -> int:
    hidden_size = config["hidden_size"]
    num_hidden_layers = config["num_hidden_layers"]
    num_attention_heads = config["num_attention_heads"]
    num_key_value_heads = config.get("num_key_value_heads", num_attention_heads)
    intermediate_size = config["intermediate_size"]
    head_dim = hidden_size // num_attention_heads
    vocab_size = config.get("vocab_size", 32000)

    q_params = hidden_size * num_attention_heads * head_dim
    k_params = hidden_size * num_key_value_heads * head_dim
    v_params = hidden_size * num_key_value_heads * head_dim
    o_params = num_attention_heads * head_dim * hidden_size
    mlp_params = 3 * hidden_size * intermediate_size
    layer_weights = q_params + k_params + v_params + o_params + mlp_params
    total_weights = num_hidden_layers * layer_weights + vocab_size * hidden_size

    bytes_per_param = config.get("bytes_per_param", 2)
    weight_bytes = total_weights * bytes_per_param

    kv_bytes_per_token = 2 * num_hidden_layers * num_key_value_heads * head_dim * bytes_per_param
    kv_bytes = batch_size * context_len * kv_bytes_per_token

    return weight_bytes + kv_bytes
