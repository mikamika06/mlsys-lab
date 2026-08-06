CONFIGS = [
    {
        "num_layers": 12,
        "hidden_size": 512,
        "moe_intermediate_size": 1024,
        "num_experts": 16,
        "num_shared_experts": 2,
        "fine_grained_factor": 2,
        "top_k": 2
    },
    {
        "num_layers": 24,
        "hidden_size": 1024,
        "moe_intermediate_size": 2048,
        "num_experts": 32,
        "num_shared_experts": 4,
        "fine_grained_factor": 4,
        "top_k": 4
    },
    {
        "num_layers": 32,
        "hidden_size": 2048,
        "moe_intermediate_size": 4096,
        "num_experts": 64,
        "num_shared_experts": 8,
        "fine_grained_factor": 8,
        "top_k": 6
    }
]

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
