def prune_real_cnn(model_profile, sparsity_ratio):
    layers = model_profile.get("layers", [])
    pruned_result = []
    for layer in layers:
        channels = layer.get("out_channels", 16)
        keep_count = max(1, int(channels * (1.0 - sparsity_ratio)))
        pruned_result.append({
            "name": layer.get("name"),
            "original_channels": channels,
            "pruned_channels": keep_count
        })
    return {"pruned_layers": pruned_result, "sparsity_ratio": sparsity_ratio}
