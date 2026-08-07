def check_marlin_eligible(cfg: dict) -> bool:
    tp_size = cfg.get("tp_size", 1)
    if not isinstance(tp_size, int) or tp_size < 1:
        return False

    mode = cfg.get("parallel_mode", "col")
    k = cfg.get("k", 0)
    n = cfg.get("n", 0)

    if mode == "col":
        if n % tp_size != 0:
            return False
        n_rank = n // tp_size
        k_rank = k
    elif mode == "row":
        if k % tp_size != 0:
            return False
        n_rank = n
        k_rank = k // tp_size
    else:
        return False

    if k_rank % 64 != 0 or n_rank % 64 != 0:
        return False

    group_size = cfg.get("group_size", -1)
    if group_size != -1:
        if group_size <= 0 or group_size % 32 != 0:
            return False
        if k_rank % group_size != 0:
            return False

    block_n = cfg.get("block_n", 128)
    block_k = cfg.get("block_k", 64)

    if block_n not in (64, 128, 256) or block_k not in (64, 128):
        return False

    if n_rank % block_n != 0 or k_rank % block_k != 0:
        return False

    if group_size != -1:
        if (group_size % block_k != 0) and (block_k % group_size != 0):
            return False

    return True
