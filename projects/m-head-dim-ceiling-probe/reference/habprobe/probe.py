def probe_head_dim_ceiling(config):
    hd = config.get("head_dim", 128)
    if hd <= 128:
        return 128
    elif hd <= 256:
        return 256
    return 512
