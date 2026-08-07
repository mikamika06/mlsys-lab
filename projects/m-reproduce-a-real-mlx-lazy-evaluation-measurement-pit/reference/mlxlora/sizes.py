def calculate_model_sizes(model_config, lora_config):
    num_layers = model_config["num_layers"]
    bits = model_config["bits"]
    group_size = model_config["group_size"]
    scale_bytes = model_config.get("scale_bytes_per_param", 2)
    non_quant_bytes = model_config.get("non_quantized_bytes_per_param", 2)

    layer_modules = model_config["modules_per_layer"]
    other_modules = model_config["other_modules"]

    base_bytes = 0
    for mod_name, shape in layer_modules.items():
        n_elem = shape[0] * shape[1]
        weight_bytes = (n_elem * bits) // 8
        groups = n_elem // group_size
        scale_bias_bytes = groups * scale_bytes
        mod_bytes = weight_bytes + scale_bias_bytes
        base_bytes += mod_bytes * num_layers

    for mod_name, shape in other_modules.items():
        n_elem = shape[0] if len(shape) == 1 else shape[0] * shape[1]
        base_bytes += n_elem * non_quant_bytes

    r = lora_config["r"]
    target_modules = lora_config["target_modules"]
    adapter_bytes_per_param = lora_config.get("adapter_bytes_per_param", 2)
    header_bytes = lora_config.get("safetensors_header_bytes", 1024)

    adapter_bytes = 0
    for mod_name in target_modules:
        if mod_name in layer_modules:
            in_dim, out_dim = layer_modules[mod_name]
            params = (in_dim * r) + (r * out_dim)
            adapter_bytes += params * adapter_bytes_per_param * num_layers

    adapter_bytes += header_bytes
    ratio = adapter_bytes / base_bytes

    return {
        "base_bytes": base_bytes,
        "adapter_bytes": adapter_bytes,
        "ratio": ratio,
        "adapter_percentage": ratio * 100.0
    }
