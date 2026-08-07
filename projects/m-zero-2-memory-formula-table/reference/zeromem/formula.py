def compute_memory_table(param_bytes, world_size, optimizer_precision_bytes):
    model_states = param_bytes * (4 + optimizer_precision_bytes)
    gradients = param_bytes
    optimizer_states = (param_bytes * 12) // world_size
    partitioned_gradients = param_bytes // world_size
    activations = param_bytes // 4
    total_peak = model_states + gradients + optimizer_states + partitioned_gradients + activations
    return {
        "model_states": model_states,
        "gradients": gradients,
        "optimizer_states": optimizer_states,
        "partitioned_gradients": partitioned_gradients,
        "activations": activations,
        "total_peak": total_peak
    }
