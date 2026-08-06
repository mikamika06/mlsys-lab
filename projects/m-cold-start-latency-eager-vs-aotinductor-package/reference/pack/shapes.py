def derive_shape_assertions(config):
    return {
        "min_seq": config["min_seq"],
        "max_seq": config["max_seq"],
        "hidden_dim": config["hidden_dim"],
        "valid": True
    }
