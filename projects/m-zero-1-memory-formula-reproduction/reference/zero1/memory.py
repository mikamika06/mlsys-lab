def calculate_zero1_memory(num_params, world_size, precision_bytes=2):
    fp32_bytes = 4
    param_bytes = num_params * precision_bytes
    grad_bytes = num_params * precision_bytes
    opt_state_total_bytes = num_params * (fp32_bytes + fp32_bytes + fp32_bytes)

    baseline_bytes = param_bytes + grad_bytes + opt_state_total_bytes
    opt_state_per_rank = opt_state_total_bytes / world_size
    zero1_bytes = param_bytes + grad_bytes + opt_state_per_rank

    return {
        "baseline_bytes": float(baseline_bytes),
        "zero1_bytes": float(zero1_bytes),
        "opt_state_per_rank_bytes": float(opt_state_per_rank),
    }
