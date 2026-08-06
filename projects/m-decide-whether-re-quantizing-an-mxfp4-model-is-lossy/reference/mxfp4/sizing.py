def compute_mxfp4_share(config):
    num_params = config.get("total_params", 1000000)
    bs = config.get("block_size", 16)
    weight_bits = 4.0
    scale_bits = 8.0
    bytes_per_weight = weight_bits / 8.0
    bytes_per_scale = (scale_bits / 8.0) / bs
    mxfp4_bytes = num_params * (bytes_per_weight + bytes_per_scale)
    other_bytes = num_params * 0.2 * 2.0
    total_bytes = mxfp4_bytes + other_bytes
    return float(mxfp4_bytes / total_bytes)
