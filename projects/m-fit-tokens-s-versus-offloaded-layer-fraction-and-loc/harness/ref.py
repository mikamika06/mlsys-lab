CONFIGS = [
    {
        "total_layers": 32,
        "base_overhead_bytes": 500 * 1024 * 1024,
        "bytes_per_layer_weight": 100 * 1024 * 1024,
        "bytes_per_layer_kv": 20 * 1024 * 1024,
    },
    {
        "total_layers": 60,
        "base_overhead_bytes": 1000 * 1024 * 1024,
        "bytes_per_layer_weight": 250 * 1024 * 1024,
        "bytes_per_layer_kv": 50 * 1024 * 1024,
    },
    {
        "total_layers": 16,
        "base_overhead_bytes": 200 * 1024 * 1024,
        "bytes_per_layer_weight": 50 * 1024 * 1024,
        "bytes_per_layer_kv": 10 * 1024 * 1024,
    },
]

PROFILES = [
    {0.0: 100.0, 0.2: 95.0, 0.4: 90.0, 0.6: 25.0, 0.8: 10.0, 1.0: 2.0},
    {0.0: 50.0, 0.25: 48.0, 0.50: 45.0, 0.75: 42.0, 1.0: 8.0},
    {0.0: 120.0, 0.33: 110.0, 0.66: 30.0, 1.0: 5.0},
]


def find_offload_cliff(profiles):
    sorted_fracs = sorted(profiles.keys())
    if not sorted_fracs:
        return 0.0
    for i in range(1, len(sorted_fracs)):
        prev_f = sorted_fracs[i - 1]
        curr_f = sorted_fracs[i]
        prev_tok = profiles[prev_f]
        curr_tok = profiles[curr_f]
        if prev_tok > 0 and (prev_tok - curr_tok) / prev_tok >= 0.30:
            return curr_f
    return 1.0


def fit_layers_in_budget(model_config, memory_budget_bytes):
    base = model_config["base_overhead_bytes"]
    if memory_budget_bytes < base:
        return 0
    avail = memory_budget_bytes - base
    cost_per_layer = model_config["bytes_per_layer_weight"] + model_config["bytes_per_layer_kv"]
    if cost_per_layer <= 0:
        return model_config["total_layers"]
    num_layers = avail // cost_per_layer
    return min(int(num_layers), model_config["total_layers"])


def select_optimal_offload(model_config, memory_budget_bytes, profiles):
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
