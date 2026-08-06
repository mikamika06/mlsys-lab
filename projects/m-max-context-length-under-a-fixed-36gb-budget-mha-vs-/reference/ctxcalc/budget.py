def compute_token_bytes(cfg):
    if cfg["type"] in ("mha", "gqa"):
        return 2 * cfg["num_kv_heads"] * cfg["head_dim"] * cfg["bytes_per_elem"]
    elif cfg["type"] == "mla":
        return (cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"]) * cfg["bytes_per_elem"]
    raise ValueError("unknown type")


def max_context_length(cfg, budget_bytes=36 * 1024 * 1024 * 1024):
    tb = compute_token_bytes(cfg)
    total_per_token = cfg["num_layers"] * tb
    return int(budget_bytes // total_per_token)
