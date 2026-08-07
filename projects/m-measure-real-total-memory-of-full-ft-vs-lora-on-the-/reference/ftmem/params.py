def count_total_params(model_cfg):
    num_layers = model_cfg.get("num_layers", 1)
    base_params = sum(p["in_dim"] * p["out_dim"] for p in model_cfg.get("projections", [])) * num_layers
    mode = model_cfg.get("mode", "full")
    if mode == "full":
        return base_params
    target_modules = set(model_cfg.get("target_modules", []))
    lora_rank = model_cfg.get("lora_rank", 0)
    adapter_params = sum(lora_rank * (p["in_dim"] + p["out_dim"]) for p in model_cfg.get("projections", []) if p["name"] in target_modules) * num_layers
    return base_params + adapter_params

def count_trainable_params(model_cfg):
    num_layers = model_cfg.get("num_layers", 1)
    mode = model_cfg.get("mode", "full")
    if mode == "full":
        return sum(p["in_dim"] * p["out_dim"] for p in model_cfg.get("projections", [])) * num_layers
    target_modules = set(model_cfg.get("target_modules", []))
    lora_rank = model_cfg.get("lora_rank", 0)
    return sum(lora_rank * (p["in_dim"] + p["out_dim"]) for p in model_cfg.get("projections", []) if p["name"] in target_modules) * num_layers
