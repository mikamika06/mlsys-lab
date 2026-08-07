def calculate_zero3_memory(num_params, bytes_per_param, dp_degree):
    optimizer_state_bytes = (12 * num_params) / dp_degree
    gradient_bytes = (2 * num_params) / dp_degree
    parameter_bytes = (2 * num_params) / dp_degree
    activation_bytes = num_params * 0.05
    total_bytes = optimizer_state_bytes + gradient_bytes + parameter_bytes + activation_bytes
    return float(total_bytes)
