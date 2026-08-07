def validate_architecture_match(cfg, target):
    return {
        "depth_only": {"layers": cfg["num_hidden_layers"] // 2, "hidden_size": cfg["hidden_size"]},
        "width_only": {"layers": cfg["num_hidden_layers"], "hidden_size": cfg["hidden_size"] // 2},
        "params": cfg["hidden_size"] * cfg["num_hidden_layers"]
    }
