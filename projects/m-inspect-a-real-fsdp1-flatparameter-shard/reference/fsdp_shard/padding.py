import math

def compute_padding_overhead(param_sizes, world_size):
    """Compute padding overhead ratio for flat parameter sharding."""
    total = sum(param_sizes)
    padded_total = math.ceil(total / world_size) * world_size
    padding = padded_total - total
    return float(padding) / float(padded_total) if padded_total > 0 else 0.0
