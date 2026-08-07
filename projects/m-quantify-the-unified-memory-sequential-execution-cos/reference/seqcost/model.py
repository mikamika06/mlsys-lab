def model_bytes(config, bytes_per_param=2):
    h = config["hidden_size"]
    l = config["num_layers"]
    v = config["vocab_size"]
    attn_weights = l * (4 * h * h)
    mlp_weights = l * (3 * h * (4 * h))
    embed_weights = 2 * v * h
    total_params = attn_weights + mlp_weights + embed_weights
    return total_params * bytes_per_param
