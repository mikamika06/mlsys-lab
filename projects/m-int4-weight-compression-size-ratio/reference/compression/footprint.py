def memory_footprint(weights_count, bits, group_size, scale_bits):
    packed_bytes = weights_count * bits / 8.0
    num_groups = weights_count / group_size
    meta_bytes = num_groups * (scale_bits / 8.0)
    return float(packed_bytes + meta_bytes)
