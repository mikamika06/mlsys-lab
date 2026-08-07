CONFIGS = [
    {"num_params": 175_000_000_000, "bytes_per_param": 4, "dp_degree": 64},
    {"num_params": 70_000_000_000, "bytes_per_param": 2, "dp_degree": 32},
    {"num_params": 13_000_000_000, "bytes_per_param": 2, "dp_degree": 16},
]

LAYER_CONFIGS = [
    [1000000, 2000000, 500000],
    [500000, 500000, 1000000, 2000000],
    [1048576, 2097152]
]

def calculate_zero3_memory(num_params, bytes_per_param, dp_degree):
    optimizer_state_bytes = (12 * num_params) / dp_degree
    gradient_bytes = (2 * num_params) / dp_degree
    parameter_bytes = (2 * num_params) / dp_degree
    activation_bytes = num_params * 0.05
    total_bytes = optimizer_state_bytes + gradient_bytes + parameter_bytes + activation_bytes
    return float(total_bytes)

def simulate_all_gather_free_cycle(layer_sizes, dp_degree):
    timeline = []
    current_mem = 0
    peak_mem = 0
    for idx, size in enumerate(layer_sizes):
        gathered_size = size
        current_mem += gathered_size
        if current_mem > peak_mem:
            peak_mem = current_mem
        timeline.append({"layer": idx, "action": "all_gather", "memory": current_mem})
        current_mem -= (size - (size / dp_degree))
        timeline.append({"layer": idx, "action": "free", "memory": current_mem})
    return {"timeline": timeline, "peak_memory": float(peak_mem)}

def calculate_communication_volume(num_params, bytes_per_param, dp_degree):
    psi = num_params * bytes_per_param
    forward_volume = 2.0 * psi * ((dp_degree - 1.0) / dp_degree)
    backward_volume = 4.0 * psi * ((dp_degree - 1.0) / dp_degree)
    total_volume = forward_volume + backward_volume
    return float(total_volume)
