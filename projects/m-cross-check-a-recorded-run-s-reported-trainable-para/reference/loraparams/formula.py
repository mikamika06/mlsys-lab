def calculate_trainable_params(model_config: dict, lora_config: dict) -> dict:
    """Calculates theoretical trainable parameters for a model and LoRA config."""
    r = lora_config.get("r", 0)
    targets = set(lora_config.get("target_modules", []))
    bias_setting = lora_config.get("bias", "none")
    modules_to_save = set(lora_config.get("modules_to_save") or [])

    modules = model_config.get("modules", {})

    lora_params = 0
    for t in targets:
        if t in modules:
            m = modules[t]
            count = m.get("count", 1)
            lora_params += count * r * (m["in_dim"] + m["out_dim"])

    bias_params = 0
    if bias_setting == "all":
        for m in modules.values():
            if m.get("has_bias", False):
                count = m.get("count", 1)
                bias_params += count * m["out_dim"]
    elif bias_setting == "lora_only":
        for t in targets:
            if t in modules:
                m = modules[t]
                if m.get("has_bias", False):
                    count = m.get("count", 1)
                    bias_params += count * m["out_dim"]

    save_params = 0
    for s in modules_to_save:
        if s in modules:
            m = modules[s]
            count = m.get("count", 1)
            weight_p = count * m["in_dim"] * m["out_dim"]
            bias_p = (count * m["out_dim"]) if m.get("has_bias", False) else 0
            save_params += weight_p + bias_p

    total = lora_params + bias_params + save_params

    return {
        "lora_adapter_params": lora_params,
        "bias_params": bias_params,
        "modules_to_save_params": save_params,
        "total_trainable_params": total,
    }
