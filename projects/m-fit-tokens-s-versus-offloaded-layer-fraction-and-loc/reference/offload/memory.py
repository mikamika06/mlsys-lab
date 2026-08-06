def fit_layers_in_budget(model_config, memory_budget_bytes):
    """Calculate maximum layers fitting inside memory_budget_bytes.

    model_config contains:
      - base_overhead_bytes: fixed memory (CUDA context, vocab, activations)
      - bytes_per_layer_weight: static weight size per layer
      - bytes_per_layer_kv: KV cache size per layer for max_seq_len
      - total_layers: total model layers
    """
    base = model_config["base_overhead_bytes"]
    if memory_budget_bytes < base:
        return 0

    avail = memory_budget_bytes - base
    cost_per_layer = model_config["bytes_per_layer_weight"] + model_config["bytes_per_layer_kv"]

    if cost_per_layer <= 0:
        return model_config["total_layers"]

    num_layers = avail // cost_per_layer
    return min(int(num_layers), model_config["total_layers"])
