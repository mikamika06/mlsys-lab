def count_reduce_scatters(num_steps, accum_steps, use_no_sync):
    """Count reduce-scatter operations."""
    if use_no_sync:
        return num_steps // accum_steps + (1 if num_steps % accum_steps != 0 else 0)
    else:
        return num_steps
