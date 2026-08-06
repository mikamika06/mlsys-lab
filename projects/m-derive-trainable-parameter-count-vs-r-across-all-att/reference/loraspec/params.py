def count_trainable_parameters(model_config, target_modules, r):
    num_layers = model_config["num_hidden_layers"]
    hidden_size = model_config["hidden_size"]
    num_heads = model_config["num_attention_heads"]
    num_kv_heads = model_config.get("num_key_value_heads", num_heads)
    intermediate_size = model_config["intermediate_size"]
    head_dim = model_config.get("head_dim", hidden_size // num_heads)

    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    dim_map = {
        "q_proj": (hidden_size, q_dim),
        "k_proj": (hidden_size, kv_dim),
        "v_proj": (hidden_size, kv_dim),
        "o_proj": (q_dim, hidden_size),
        "gate_proj": (hidden_size, intermediate_size),
        "up_proj": (hidden_size, intermediate_size),
        "down_proj": (intermediate_size, hidden_size),
    }

    total_per_layer = 0
    for mod in target_modules:
        if mod in dim_map:
            d_in, d_out = dim_map[mod]
            total_per_layer += r * (d_in + d_out)

    return total_per_layer * num_layers
