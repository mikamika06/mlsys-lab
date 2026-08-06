def compute_arithmetic_intensity(batch_size, hidden_size, num_layers):
    params_per_layer = 4 * hidden_size * hidden_size
    total_params = num_layers * params_per_layer
    bytes_loaded = total_params * 2
    bytes_transferred = bytes_loaded + batch_size * hidden_size * 2 * 2
    flops = 2 * total_params * batch_size
    return flops / bytes_transferred
