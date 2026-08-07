def compute_zero2_memory(model_params_bytes, num_ranks, optimizer_element_size=4):
    psi = model_params_bytes
    N = num_ranks
    k = optimizer_element_size
    optimizer_states = (12.0 * psi) / N
    gradients = psi
    parameters = psi
    total = optimizer_states + gradients + parameters
    return {
        "optimizer_states": float(optimizer_states),
        "gradients": float(gradients),
        "parameters": float(parameters),
        "total": float(total)
    }
