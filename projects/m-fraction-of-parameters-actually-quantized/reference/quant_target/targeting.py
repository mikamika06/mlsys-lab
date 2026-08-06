def filter_target_modules(config, ignore_list=None):
    ignore_list = ignore_list or []
    layers = config.get("layers", [])
    is_multimodal = config.get("is_multimodal", False)
    language_only = config.get("language_only_quant", True)
    targeted = []
    for layer in layers:
        name = layer.get("name", "")
        mod_type = layer.get("type", "")
        if is_multimodal and language_only and "vision" in name.lower():
            continue
        if any(ig in name for ig in ignore_list):
            continue
        targeted.append({"name": name, "type": mod_type, "params": layer.get("params", 0)})
    if not any(l.get("name") == "lm_head" for l in targeted):
        if not any(ig == "lm_head" for ig in ignore_list):
            lm_layer = next((l for l in layers if l.get("name") == "lm_head"), None)
            if lm_layer:
                targeted.append(lm_layer)
    return targeted
