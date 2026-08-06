def compute_quantized_fraction(config, target_modules):
    layers = config.get("layers", [])
    total_params = sum(l.get("params", 0) for l in layers)
    if total_params == 0:
        return 0.0, 0
    quant_params = sum(t.get("params", 0) for t in target_modules)
    fraction = quant_params / total_params
    lm_head_layer = next((l for l in layers if l.get("name") == "lm_head"), None)
    head_cost = lm_head_layer.get("params", 0) if lm_head_layer else 0
    return float(fraction), int(head_cost)
