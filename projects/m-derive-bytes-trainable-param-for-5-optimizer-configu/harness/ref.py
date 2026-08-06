CONFIGS = [
    {"name": "sgd_fp32", "weight_bytes": 4, "grad_bytes": 4, "optimizer_states": []},
    {"name": "adamw_fp32", "weight_bytes": 4, "grad_bytes": 4, "optimizer_states": [{"name": "m", "bytes_per_elem": 4}, {"name": "v", "bytes_per_elem": 4}]},
    {"name": "adamw_mixed", "weight_bytes": 4, "grad_bytes": 4, "optimizer_states": [{"name": "m", "bytes_per_elem": 4}, {"name": "v", "bytes_per_elem": 4}]},
    {"name": "adamw_fp16", "weight_bytes": 2, "grad_bytes": 2, "optimizer_states": [{"name": "m", "bytes_per_elem": 4}, {"name": "v", "bytes_per_elem": 4}, {"name": "w_fp32", "bytes_per_elem": 4}]},
    {"name": "adamw_8bit", "weight_bytes": 4, "grad_bytes": 4, "optimizer_states": [{"name": "m", "bytes_per_elem": 1}, {"name": "v", "bytes_per_elem": 1}]}
]


def bytes_per_param(cfg):
    w = cfg.get("weight_bytes", 4)
    g = cfg.get("grad_bytes", 4)
    s = sum(st.get("bytes_per_elem", 4) for st in cfg.get("optimizer_states", []))
    return w + g + s


def total_memory(params_bytes, cfg):
    num = params_bytes / 4
    return num * bytes_per_param(cfg)


def gap(full, lora):
    if lora == 0:
        return 0.0
    return float(full) / float(lora)
