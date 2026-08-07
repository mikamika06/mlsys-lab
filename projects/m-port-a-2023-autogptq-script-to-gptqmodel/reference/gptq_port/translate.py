def translate_config(legacy_cfg):
    cfg = dict(legacy_cfg)
    new_cfg = {}
    new_cfg["quant_method"] = "gptq"
    new_cfg["bits"] = cfg.get("bits", 4)
    new_cfg["group_size"] = cfg.get("group_size", 128)
    new_cfg["damp_percent"] = cfg.get("damp_percent", 0.1)
    new_cfg["desc_act"] = cfg.get("desc_act", False)
    new_cfg["static_groups"] = cfg.get("static_groups", False)
    new_cfg["sym"] = cfg.get("sym", True)
    new_cfg["true_sequential"] = cfg.get("true_sequential", True)
    new_cfg["model_name_or_path"] = cfg.get("model_name_or_path", None)
    new_cfg["model_file_base_name"] = cfg.get("model_file_base_name", "model")
    return new_cfg
