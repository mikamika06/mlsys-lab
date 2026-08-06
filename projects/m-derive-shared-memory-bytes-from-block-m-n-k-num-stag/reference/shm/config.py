def reconstruct_configs(
    error_msg: str, max_bytes: int, candidates: list
) -> list:
    valid = []
    for cfg in candidates:
        b = cfg.get("block_m", 64)
        n = cfg.get("block_n", 64)
        k = cfg.get("block_k", 32)
        s = cfg.get("num_stages", 3)
        dt = cfg.get("dtype", "float16")
        from shm.bytes import compute_shm_bytes

        if compute_shm_bytes(b, n, k, s, dt) <= max_bytes:
            valid.append(cfg)
    return valid
