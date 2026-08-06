def count_parameters(config):
    """Count model and trainable parameters."""
    L = config["num_layers"]
    h = config["hidden_dim"]
    r = config["lora_rank"]
    weight_per_layer = 4 * h * h
    total_base = L * weight_per_layer
    lora_per_layer = 2 * h * r + 2 * r * h
    total_lora_trainable = L * lora_per_layer
    return {
        "total_base": total_base,
        "full_trainable": total_base,
        "lora_trainable": total_lora_trainable,
        "lora_frozen": total_base,
    }
