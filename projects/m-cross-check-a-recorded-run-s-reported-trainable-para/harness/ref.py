MODEL_CONFIGS = [
    {
        "modules": {
            "q_proj": {"in_dim": 4096, "out_dim": 4096, "count": 32, "has_bias": False},
            "v_proj": {"in_dim": 4096, "out_dim": 4096, "count": 32, "has_bias": False},
            "k_proj": {"in_dim": 4096, "out_dim": 4096, "count": 32, "has_bias": False},
            "o_proj": {"in_dim": 4096, "out_dim": 4096, "count": 32, "has_bias": False},
            "gate_proj": {"in_dim": 4096, "out_dim": 11008, "count": 32, "has_bias": False},
            "up_proj": {"in_dim": 4096, "out_dim": 11008, "count": 32, "has_bias": False},
            "down_proj": {"in_dim": 11008, "out_dim": 4096, "count": 32, "has_bias": False},
            "embed_tokens": {"in_dim": 32000, "out_dim": 4096, "count": 1, "has_bias": False},
            "lm_head": {"in_dim": 4096, "out_dim": 32000, "count": 1, "has_bias": False},
        }
    },
    {
        "modules": {
            "q_proj": {"in_dim": 2048, "out_dim": 2048, "count": 16, "has_bias": True},
            "v_proj": {"in_dim": 2048, "out_dim": 2048, "count": 16, "has_bias": True},
            "dense": {"in_dim": 2048, "out_dim": 8192, "count": 16, "has_bias": True},
        }
    },
    {
        "modules": {
            "query": {"in_dim": 768, "out_dim": 768, "count": 12, "has_bias": False},
            "key": {"in_dim": 768, "out_dim": 768, "count": 12, "has_bias": False},
            "value": {"in_dim": 768, "out_dim": 768, "count": 12, "has_bias": False},
            "cls_head": {"in_dim": 768, "out_dim": 10, "count": 1, "has_bias": True},
        }
    },
    {
        "modules": {
            "c_attn": {"in_dim": 1024, "out_dim": 3072, "count": 24, "has_bias": True},
            "c_proj": {"in_dim": 1024, "out_dim": 1024, "count": 24, "has_bias": True},
        }
    },
    {
        "modules": {
            "in_layer": {"in_dim": 512, "out_dim": 512, "count": 8, "has_bias": False},
            "out_layer": {"in_dim": 512, "out_dim": 512, "count": 8, "has_bias": False},
        }
    },
]

LORA_CONFIGS = [
    {"r": 8, "target_modules": ["q_proj", "v_proj"], "bias": "none"},
    {"r": 16, "target_modules": ["q_proj", "v_proj"], "bias": "lora_only"},
    {"r": 64, "target_modules": ["query", "value"], "modules_to_save": ["cls_head"], "bias": "all"},
    {"r": 32, "target_modules": ["c_attn"], "bias": "none"},
    {"r": 4, "target_modules": ["in_layer", "out_layer"], "bias": "none"},
]


def ref_calculate_trainable_params(model_config, lora_config):
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


def get_sample_runs():
    runs = []
    for i in range(5):
        m_cfg = MODEL_CONFIGS[i]
        l_cfg = LORA_CONFIGS[i]
        expected = ref_calculate_trainable_params(m_cfg, l_cfg)["total_trainable_params"]
        reported = expected if i not in (1, 3) else expected + (10000 * (i + 1))
        runs.append({
            "run_id": f"run-100{i}",
            "model_config": m_cfg,
            "lora_config": l_cfg,
            "reported_trainable_params": reported,
        })
    return runs
