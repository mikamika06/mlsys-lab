def predict_fsdp_units(layer_sizes, threshold):
    units = 0
    current_size = 0
    for size in layer_sizes:
        if current_size + size > threshold and current_size > 0:
            units += 1
            current_size = size
        else:
            current_size += size
    if current_size > 0:
        units += 1
    return int(units)
