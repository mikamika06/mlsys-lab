def compare_zero3_dp_comm(params, world_size, bytes_per_param):
    factor = (world_size - 1) / world_size
    dp_bytes = 2.0 * factor * params * bytes_per_param
    zero3_bytes = factor * params * bytes_per_param + factor * params * bytes_per_param
    return {
        "dp_bytes": float(dp_bytes),
        "zero3_bytes": float(zero3_bytes),
        "ratio": float(zero3_bytes / dp_bytes),
    }
