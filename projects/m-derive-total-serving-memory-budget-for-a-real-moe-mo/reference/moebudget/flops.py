def compute_crossover_context(cfg):
    h = cfg["hidden_size"]
    l = cfg["num_layers"]
    kv_heads = cfg["num_kv_heads"]
    hd = cfg["head_dim"]
    n_exp = cfg["num_experts"]
    a_exp = cfg["active_experts"]
    exp_h = cfg["expert_hidden_size"]
    attn_flops_per_token = 4 * l * h
    ffn_flops_per_token = 2 * l * a_exp * (3 * h * exp_h)
    ratio = ffn_flops_per_token / (2 * l * kv_heads * hd)
    return float(ratio)
