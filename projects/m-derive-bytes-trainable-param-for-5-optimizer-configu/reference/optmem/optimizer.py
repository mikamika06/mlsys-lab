def bytes_per_trainable_param(optimizer_config_name):
    mapping = {
        "adamw_pure_fp32": 16.0,
        "adamw_mixed_precision": 16.0,
        "sgd_momentum_fp32": 12.0,
        "adagrad_fp32": 12.0,
        "adafactor_factored": 6.0,
    }
    if optimizer_config_name not in mapping:
        raise ValueError(f"Unknown optimizer config: {optimizer_config_name}")
    return mapping[optimizer_config_name]
