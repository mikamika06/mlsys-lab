def compute_communication_volumes(num_params_elements, bytes_per_element, world_size):
    total_bytes = num_params_elements * bytes_per_element
    zero1_vol = 2.0 * total_bytes * (world_size - 1) / world_size
    zero3_vol = 2.0 * total_bytes * (world_size - 1) / world_size + total_bytes
    return {"zero1": zero1_vol, "zero3": zero3_vol}
