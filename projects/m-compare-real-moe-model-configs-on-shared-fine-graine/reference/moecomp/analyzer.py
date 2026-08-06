def parse_config(cfg):
    return {
        "num_layers": cfg.get("num_layers", 0),
        "hidden_size": cfg.get("hidden_size", 0),
        "moe_intermediate_size": cfg.get("moe_intermediate_size", 0),
        "num_experts": cfg.get("num_experts", 0),
        "num_shared_experts": cfg.get("num_shared_experts", 0),
        "fine_grained_factor": cfg.get("fine_grained_factor", 1),
        "top_k": cfg.get("top_k", 1)
    }
