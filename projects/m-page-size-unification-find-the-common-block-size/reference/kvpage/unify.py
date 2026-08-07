def find_common_block_size(config, candidate_block_sizes=(8, 16, 32, 64), page_align_bytes=64):
    for b in candidate_block_sizes:
        valid = True
        for layer in config["layers"]:
            kv_heads = layer["kv_heads"]
            head_dim = layer["head_dim"]
            dtype_bytes = layer.get("dtype_bytes", 2)
            bytes_per_token = 2 * kv_heads * head_dim * dtype_bytes
            block_bytes = b * bytes_per_token
            if block_bytes % page_align_bytes != 0:
                valid = False
                break
        if valid:
            return b
    return candidate_block_sizes[-1]
