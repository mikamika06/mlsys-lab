def compute_vram_budget(model_config, vram_limit_bytes, offload_experts):
    non_exp = model_config["non_expert_bytes"]
    n_layers = model_config["layers"]
    n_exp = model_config["num_experts"]
    exp_size = model_config["expert_size_bytes"]
    resident_exp_bytes = 0 if offload_experts else (n_layers * n_exp * exp_size)
    return non_exp + resident_exp_bytes

def max_context_length(model_config, vram_limit_bytes, offload_experts):
    fixed = compute_vram_budget(model_config, vram_limit_bytes, offload_experts)
    available = vram_limit_bytes - fixed
    if available <= 0:
        return 0
    kv_bytes_per_token = model_config["kv_bytes_per_token"]
    return int(available // kv_bytes_per_token)
