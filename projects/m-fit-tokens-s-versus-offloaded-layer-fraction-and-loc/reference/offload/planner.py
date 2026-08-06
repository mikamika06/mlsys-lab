from offload.memory import fit_layers_in_budget


def select_optimal_offload(model_config, memory_budget_bytes, profiles):
    """Select num_gpu_layers maximizing tok/s under memory constraints."""
    max_gpu_layers = fit_layers_in_budget(model_config, memory_budget_bytes)
    total_layers = model_config["total_layers"]

    best_num_gpu = 0
    best_tok_s = -1.0

    for num_gpu in range(max_gpu_layers + 1):
        offloaded_frac = round((total_layers - num_gpu) / total_layers, 4)

        closest_frac = min(profiles.keys(), key=lambda f: abs(f - offloaded_frac))
        tok_s = profiles[closest_frac]

        if tok_s > best_tok_s:
            best_tok_s = tok_s
            best_num_gpu = num_gpu

    return best_num_gpu
