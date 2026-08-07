def calculate_communication_volume(num_params, bytes_per_param, dp_degree):
    psi = num_params * bytes_per_param
    forward_volume = 2.0 * psi * ((dp_degree - 1.0) / dp_degree)
    backward_volume = 4.0 * psi * ((dp_degree - 1.0) / dp_degree)
    total_volume = forward_volume + backward_volume
    return float(total_volume)
