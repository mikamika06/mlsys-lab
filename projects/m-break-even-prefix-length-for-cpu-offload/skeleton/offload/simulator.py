def compute_kv_bytes(config, prefix_len):
    raise NotImplementedError


def estimate_recompute_time(config, hw, prefix_len):
    raise NotImplementedError


def estimate_load_time(config, tier, prefix_len):
    raise NotImplementedError


def compute_breakeven_prefix_length(config, hw, tier):
    raise NotImplementedError
