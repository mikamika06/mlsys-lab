def compute_dense_vs_hybrid_cost(config, context_length, dtype_bytes=2):
    dense_bytes = 0
    hybrid_bytes = 0
    for layer in config["layers"]:
        if layer["kind"] in ("full", "sliding"):
            kv_heads = layer["kv_heads"]
            head_dim = layer["head_dim"]
            layer_bytes = 2 * context_length * kv_heads * head_dim * dtype_bytes
            dense_bytes += layer_bytes
            hybrid_bytes += layer_bytes
        elif layer["kind"] == "mamba":
            state_dim = layer["state_dim"]
            d_inner = layer["d_inner"]
            s_bytes = 2 * d_inner * state_dim * dtype_bytes
            hybrid_bytes += s_bytes
            dense_bytes += 2 * context_length * 8 * 128 * dtype_bytes
    return {"dense_bytes": dense_bytes, "hybrid_bytes": hybrid_bytes}
