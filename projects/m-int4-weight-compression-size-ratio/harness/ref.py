CONFIGS = [
    {"weights_count": 1024 * 1024 * 768, "bits": 4, "group_size": 128, "scale_bits": 16},
    {"weights_count": 1024 * 1024 * 512, "bits": 4, "group_size": 64, "scale_bits": 16},
    {"weights_count": 1024 * 1024 * 1024, "bits": 8, "group_size": 128, "scale_bits": 16},
    {"weights_count": 1024 * 1024 * 256, "bits": 4, "group_size": 32, "scale_bits": 32},
    {"weights_count": 1024 * 1024 * 2048, "bits": 4, "group_size": 256, "scale_bits": 16},
]

def compute_size_ratio(weights_count, bits, group_size, scale_bits):
    orig_bits = 16
    orig_bytes = weights_count * orig_bits / 8.0
    num_groups = weights_count / group_size
    packed_bytes = weights_count * bits / 8.0
    meta_bytes = num_groups * (scale_bits / 8.0)
    total_compressed_bytes = packed_bytes + meta_bytes
    return float(orig_bytes / total_compressed_bytes)

def compute_footprint(weights_count, bits, group_size, scale_bits):
    packed_bytes = weights_count * bits / 8.0
    num_groups = weights_count / group_size
    meta_bytes = num_groups * (scale_bits / 8.0)
    return float(packed_bytes + meta_bytes)

def compute_perplexity_delta(data_free_ppl, awq_ppl):
    return float(data_free_ppl - awq_ppl)
