def bytes_per_param(optimizer_config):
    w_bytes = optimizer_config.get("weight_bytes", 4)
    g_bytes = optimizer_config.get("grad_bytes", 4)
    opt_states = optimizer_config.get("optimizer_states", [])
    total = w_bytes + g_bytes
    for st in opt_states:
        total += st.get("bytes_per_elem", 4)
    return total
