def estimate_checkpoint_size(config, scheme):
    bits = scheme.get("bits", 16)
    group_size = scheme.get("group_size", 128)
    total_weights = config.get("total_weight_params", 0)
    meta_overhead_factor = scheme.get("meta_overhead_factor", 1.02)
    raw_bytes = total_weights * (bits / 8.0)
    if bits < 16:
        scales_and_zeros_ratio = 16.0 / group_size
        raw_bytes += total_weights * (scales_and_zeros_ratio * (16.0 / 8.0))
    return int(raw_bytes * meta_overhead_factor)
