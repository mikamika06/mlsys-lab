CONFIGS = [
    {
        "is_multimodal": True,
        "language_only_quant": True,
        "layers": [
            {"name": "vision.encoder", "type": "linear", "params": 4000},
            {"name": "model.layer.0.attn", "type": "linear", "params": 3000},
            {"name": "lm_head", "type": "linear", "params": 3000}
        ]
    },
    {
        "is_multimodal": False,
        "language_only_quant": False,
        "layers": [
            {"name": "model.layer.0.attn", "type": "linear", "params": 5000},
            {"name": "model.layer.1.attn", "type": "linear", "params": 5000}
        ]
    },
    {
        "is_multimodal": True,
        "language_only_quant": True,
        "layers": [
            {"name": "vision_tower.patch_embed", "type": "conv", "params": 2000},
            {"name": "model.decoder.layers.0.mlp", "type": "linear", "params": 4000},
            {"name": "lm_head", "type": "linear", "params": 4000}
        ]
    }
]

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
