def compute_costs(cfg):
    hs = cfg.get("hidden_size", 0)
    inter = cfg.get("moe_intermediate_size", 0)
    num_exp = cfg.get("num_experts", 0)
    num_shared = cfg.get("num_shared_experts", 0)
    fg_factor = cfg.get("fine_grained_factor", 1)
    effective_experts = num_exp // fg_factor
    shared_params = num_shared * 2 * hs * inter
    routed_params = effective_experts * 2 * hs * (inter // fg_factor)
    total_params = shared_params + routed_params
    return {
        "shared_params": shared_params,
        "routed_params": routed_params,
        "total_params": total_params,
        "effective_experts": effective_experts
    }
