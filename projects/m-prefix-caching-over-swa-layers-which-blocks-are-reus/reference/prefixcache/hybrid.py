def compute_mamba_state_size(layer, dtype_bytes=2):
    if layer["kind"] != "mamba":
        return 0
    state_dim = layer["state_dim"]
    d_inner = layer["d_inner"]
    return 2 * d_inner * state_dim * dtype_bytes
