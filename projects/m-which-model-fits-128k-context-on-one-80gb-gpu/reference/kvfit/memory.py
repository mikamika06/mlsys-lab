def kv_cache_bytes(specs, context_len, bits):
    bytes_per_elem = bits / 8.0
    return int(2 * specs["layers"] * specs["kv_heads"] * specs["head_dim"] * context_len * bytes_per_elem)
