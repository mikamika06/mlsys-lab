import math


def compute_paged_waste(length_histogram, block_size):
    """Compute total wasted token slots in paged allocation across a length histogram."""
    total_waste = 0
    for length, count in length_histogram.items():
        if length == 0:
            continue
        allocated_tokens = math.ceil(length / block_size) * block_size
        waste_per_seq = allocated_tokens - length
        total_waste += waste_per_seq * count
    return total_waste


def compute_contiguous_waste(length_histogram, max_possible_len):
    """Compute wasted token slots assuming worst-case static contiguous reservation."""
    total_waste = 0
    for length, count in length_histogram.items():
        waste_per_seq = max_possible_len - length
        total_waste += waste_per_seq * count
    return total_waste


def compute_waste_ratio(length_histogram, block_size, max_possible_len):
    """Compute ratio of contiguous waste to paged waste."""
    paged = compute_paged_waste(length_histogram, block_size)
    contiguous = compute_contiguous_waste(length_histogram, max_possible_len)
    if paged == 0:
        return float("inf") if contiguous > 0 else 1.0
    return contiguous / paged
