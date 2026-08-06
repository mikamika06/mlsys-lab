def predict_fsdp_units(num_layers, wrap_threshold_params, layer_param_count):
    units = 0
    current_params = 0
    for _ in range(num_layers):
        if current_params + layer_param_count > wrap_threshold_params and current_params > 0:
            units += 1
            current_params = layer_param_count
        else:
            current_params += layer_param_count
    if current_params > 0:
        units += 1
    return max(1, units)
