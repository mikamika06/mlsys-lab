def compute_zero_communication_volume(num_params_bytes, world_size):
    p = float(num_params_bytes)
    n = float(world_size)
    zero1_bytes = 2.0 * p * (n - 1.0) / n
    zero3_bytes = 2.0 * p * (n - 1.0) / n + 2.0 * p
    return {"zero1_comm_bytes": float(zero1_bytes), "zero3_comm_bytes": float(zero3_bytes)}
