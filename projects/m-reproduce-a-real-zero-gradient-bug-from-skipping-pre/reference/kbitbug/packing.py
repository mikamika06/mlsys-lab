def compute_token_utilization(sample_lengths, max_length):
    total_samples = len(sample_lengths)
    padding_total_tokens = total_samples * max_length
    actual_tokens = sum(sample_lengths)

    current_bin = 0
    packed_bins = 1
    for l in sample_lengths:
        if current_bin + l + 1 > max_length:
            packed_bins += 1
            current_bin = l
        else:
            current_bin += l + 1

    packing_total_tokens = packed_bins * max_length
    padding_utilization = actual_tokens / padding_total_tokens if padding_total_tokens > 0 else 0.0
    packing_utilization_val = actual_tokens / packing_total_tokens if packing_total_tokens > 0 else 0.0
    return {
        "actual_tokens": actual_tokens,
        "padding_total": padding_total_tokens,
        "packing_total": packing_total_tokens,
        "padding_utilization": padding_utilization,
        "packing_utilization": packing_utilization_val
    }
