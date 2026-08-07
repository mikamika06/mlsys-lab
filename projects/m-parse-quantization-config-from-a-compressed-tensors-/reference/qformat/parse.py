def parse_quant_config(cfg):
    qc = cfg.get("quantization_config", {})
    fmt = qc.get("format", "")
    group = qc.get("config_groups", {}).get("group_0", {})
    w = group.get("weights", {})

    w_bits = w.get("num_bits", 8)
    group_size = w.get("group_size", 0)

    if fmt == "nvfp4":
        has_global = "weight_scales" in group and group["weight_scales"].get("group_size", 1) == 0
        return {
            "format": "nvfp4",
            "w_bits": w_bits,
            "group_size": group_size,
            "global_scale": has_global
        }
    else:
        symmetric = w.get("symmetric", True)
        a_bits = group.get("input_activations", {}).get("num_bits", 16)
        return {
            "format": f"w{w_bits}a{a_bits}",
            "w_bits": w_bits,
            "group_size": group_size,
            "symmetric": symmetric
        }
