def parse_model(config):
    head_dim = config["hidden_size"] // config["num_attention_heads"]
    return {
        "name": config["name"],
        "layers": config["num_hidden_layers"],
        "kv_heads": config["num_key_value_heads"],
        "head_dim": head_dim
    }
