def parse_quantization_config(config_dict):
    groups = config_dict.get("config_groups", {})
    if not groups:
        return {"format": "dense", "num_bits": 16, "strategy": "none", "group_size": None, "type": "float"}
    g0 = list(groups.values())[0]
    w = g0.get("weights", {})
    return {
        "format": config_dict.get("format", "quantized"),
        "num_bits": w.get("num_bits", 16),
        "strategy": w.get("strategy", "tensor"),
        "group_size": w.get("group_size"),
        "type": w.get("type", "int")
    }
