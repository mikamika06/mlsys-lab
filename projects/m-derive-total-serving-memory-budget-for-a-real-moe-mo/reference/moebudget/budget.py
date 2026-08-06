def compute_serving_memory(cfg):
    h = cfg["hidden_size"]
    l = cfg["num_layers"]
    kv_heads = cfg["num_kv_heads"]
    hd = cfg["head_dim"]
    n_exp = cfg["num_experts"]
    a_exp = cfg["active_experts"]
    exp_h = cfg["expert_hidden_size"]
    bpp = cfg["bytes_per_param"]
    ctx = cfg["context_len"]
    bs = cfg["batch_size"]
    attn_weight_params = l * (4 * h * h)
    ffn_weight_params = l * n_exp * (3 * h * exp_h)
    embed_params = cfg["vocab_size"] * h
    total_params = attn_weight_params + ffn_weight_params + embed_params
    weight_memory = total_params * bpp
    kv_cache_memory = 2 * l * bs * ctx * kv_heads * hd * bpp
    activation_memory = bs * ctx * h * 4 * bpp
    return float(weight_memory + kv_cache_memory + activation_memory)
