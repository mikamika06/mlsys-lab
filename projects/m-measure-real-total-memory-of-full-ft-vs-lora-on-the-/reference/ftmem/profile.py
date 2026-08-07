def measure_full_vs_lora(model_cfg):
    num_layers = model_cfg.get("num_layers", 1)
    base_params = sum(p["in_dim"] * p["out_dim"] for p in model_cfg.get("projections", [])) * num_layers
    target_modules = set(model_cfg.get("target_modules", []))
    lora_rank = model_cfg.get("lora_rank", 0)
    adapter_params = sum(lora_rank * (p["in_dim"] + p["out_dim"]) for p in model_cfg.get("projections", []) if p["name"] in target_modules) * num_layers
    full_bytes = base_params * 12
    lora_bytes = base_params * 2 + adapter_params * 12
    ratio = lora_bytes / full_bytes if full_bytes > 0 else 0.0
    return {
        "full_bytes": full_bytes,
        "lora_bytes": lora_bytes,
        "ratio": ratio,
    }

def measure_lora_vs_qlora(model_cfg):
    num_layers = model_cfg.get("num_layers", 1)
    base_params = sum(p["in_dim"] * p["out_dim"] for p in model_cfg.get("projections", [])) * num_layers
    target_modules = set(model_cfg.get("target_modules", []))
    lora_rank = model_cfg.get("lora_rank", 0)
    adapter_params = sum(lora_rank * (p["in_dim"] + p["out_dim"]) for p in model_cfg.get("projections", []) if p["name"] in target_modules) * num_layers
    lora_bf16_bytes = base_params * 2 + adapter_params * 12
    qlora_4bit_bytes = int(base_params * 0.5) + adapter_params * 12
    ratio = qlora_4bit_bytes / lora_bf16_bytes if lora_bf16_bytes > 0 else 0.0
    return {
        "lora_bf16_bytes": lora_bf16_bytes,
        "qlora_4bit_bytes": qlora_4bit_bytes,
        "ratio": ratio,
    }
