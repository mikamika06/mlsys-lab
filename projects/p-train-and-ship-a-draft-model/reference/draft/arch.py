def create_draft_config(target_config):
    return {
        "hidden_size": target_config["hidden_size"] // 2,
        "num_hidden_layers": max(1, target_config["num_hidden_layers"] // 4),
        "vocab_size": target_config["vocab_size"]
    }
