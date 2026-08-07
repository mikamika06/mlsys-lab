def validate_pruned_model(model_data, pruned_layers):
    if not isinstance(model_data, dict):
        return False
    tensors = model_data.get("tensors", {})
    if not tensors:
        return False
    for layer in pruned_layers:
        key = f"layer_{layer}.weight"
        if key in tensors:
            return False
    return "header" in model_data and model_data["header"] == "GGUF_VALID"
