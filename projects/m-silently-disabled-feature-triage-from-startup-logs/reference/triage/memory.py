def bakeoff_kv_memory(engine_a_config, engine_b_config, target_bytes):
    def calc(cfg):
        layers = cfg.get("layers", 32)
        kv_heads = cfg.get("kv_heads", 8)
        head_dim = cfg.get("head_dim", 128)
        dtype = cfg.get("dtype_bytes", 2)
        block_size = cfg.get("block_size", 16)
        bytes_per_token = 2 * layers * kv_heads * head_dim * dtype
        block_bytes = block_size * bytes_per_token
        num_blocks = target_bytes // block_bytes
        actual_bytes = num_blocks * block_bytes
        return {"num_blocks": num_blocks, "actual_bytes": actual_bytes}
    return {
        "engine_a": calc(engine_a_config),
        "engine_b": calc(engine_b_config)
    }
