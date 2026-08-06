def validate_config(config: dict) -> bool:
    head_dim = config.get("head_dim", 0)
    if head_dim <= 0 or head_dim % 32 != 0:
        return False
    block_size = config.get("block_size", 0)
    if block_size <= 0 or block_size % 16 != 0:
        return False
    num_heads = config.get("num_heads", 0)
    if num_heads <= 0:
        return False
    return True
