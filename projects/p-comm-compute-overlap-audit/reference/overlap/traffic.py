def compute_theoretical_traffic(model_config, world_size):
    params_bytes = model_config.get("param_bytes", 0)
    return 2.0 * params_bytes * (world_size - 1) / world_size
