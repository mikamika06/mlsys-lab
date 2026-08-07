def diagnose_noop_layers(before_model, after_model):
    noops = []
    for layer_name, b_info in before_model.items():
        if layer_name not in after_model:
            continue
        a_info = after_model[layer_name]
        b_dtype = b_info.get("dtype") if isinstance(b_info, dict) else "float32"
        a_dtype = a_info.get("dtype") if isinstance(a_info, dict) else "float32"
        is_quantized = (
            a_info.get("quantized", True) if isinstance(a_info, dict) else True
        )

        if not is_quantized or a_dtype == b_dtype:
            noops.append(layer_name)
    return sorted(noops)
