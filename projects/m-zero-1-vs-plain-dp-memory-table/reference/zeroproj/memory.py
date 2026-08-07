def compute_memory_table(model_params, world_size, dtype_bytes=4):
    total_params = sum(model_params)
    plain_model_states = total_params * dtype_bytes
    plain_gradients = total_params * dtype_bytes
    plain_optimizer_states = total_params * dtype_bytes * 12
    plain_total = plain_model_states + plain_gradients + plain_optimizer_states

    zero1_model_states = total_params * dtype_bytes
    zero1_gradients = total_params * dtype_bytes
    zero1_optimizer_states = (total_params * dtype_bytes * 12) / world_size
    zero1_total = zero1_model_states + zero1_gradients + zero1_optimizer_states

    return {
        "plain": {
            "model_states": plain_model_states,
            "gradients": plain_gradients,
            "optimizer_states": plain_optimizer_states,
            "total": plain_total,
        },
        "zero1": {
            "model_states": zero1_model_states,
            "gradients": zero1_gradients,
            "optimizer_states": zero1_optimizer_states,
            "total": zero1_total,
        },
    }
