def calculate_vram(model_config, offload_experts_to_cpu: bool):
    n_layers = model_config["n_layers"]
    hidden_dim = model_config["hidden_dim"]
    intermediate_dim = model_config["intermediate_dim"]
    n_experts = model_config["n_experts"]
    bytes_per_param = model_config.get("bytes_per_param", 2)

    non_expert_bytes_per_layer = (4 * hidden_dim * hidden_dim + 2 * hidden_dim * intermediate_dim) * bytes_per_param
    expert_bytes_per_layer = n_experts * (2 * hidden_dim * intermediate_dim + intermediate_dim * hidden_dim) * bytes_per_param

    base_vram = n_layers * non_expert_bytes_per_layer
    if not offload_experts_to_cpu:
        base_vram += n_layers * expert_bytes_per_layer

    embedding_bytes = model_config["vocab_size"] * hidden_dim * bytes_per_param
    return base_vram + embedding_bytes


def max_context_length(vram_total_bytes, model_config, offload_experts_to_cpu: bool):
    static_vram = calculate_vram(model_config, offload_experts_to_cpu)
    if static_vram >= vram_total_bytes:
        return 0
    available_for_kv = vram_total_bytes - static_vram

    n_layers = model_config["n_layers"]
    n_kv_heads = model_config["n_kv_heads"]
    head_dim = model_config["head_dim"]
    bytes_per_param = model_config.get("bytes_per_param", 2)

    kv_bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * bytes_per_param
    if kv_bytes_per_token == 0:
        return 0
    return int(available_for_kv // kv_bytes_per_token)
