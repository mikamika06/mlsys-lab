def size_ratio(weights_count, bits, group_size, scale_bits):
    orig_bytes = weights_count * 16 / 8.0
    num_groups = weights_count / group_size
    packed_bytes = weights_count * bits / 8.0
    meta_bytes = num_groups * (scale_bits / 8.0)
    total_compressed = packed_bytes + meta_bytes
    return float(orig_bytes / total_compressed)
