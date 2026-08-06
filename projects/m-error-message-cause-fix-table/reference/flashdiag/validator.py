def validate_config(config):
    head_dim = config.get("head_dim", 0)
    if head_dim <= 0 or head_dim % 32 != 0:
        return False
    block_m = config.get("block_m", 64)
    block_n = config.get("block_n", 64)
    if block_m <= 0 or block_n <= 0:
        return False
    return True
