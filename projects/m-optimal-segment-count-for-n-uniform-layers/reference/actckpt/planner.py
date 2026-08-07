import math


def optimal_segment_count(n_layers: int) -> int:
    """Returns the optimal segment count k = round(sqrt(N)) constrained to [1, N]."""
    if n_layers <= 0:
        return 1
    k = int(round(math.sqrt(n_layers)))
    return max(1, min(n_layers, k))


def analytical_peak_activation_memory(n_layers: int, num_segments: int, bytes_per_layer: int) -> int:
    """Calculates peak activation memory in bytes for N uniform layers split into num_segments."""
    if n_layers <= 0 or num_segments <= 0:
        return 0
    k = min(n_layers, num_segments)
    base_seg = n_layers // k
    rem = n_layers % k
    max_seg_size = base_seg + (1 if rem > 0 else 0)
    return (k * bytes_per_layer) + (max_seg_size * bytes_per_layer)


def analytical_recompute_ops(n_layers: int, num_segments: int, ops_per_layer: int) -> int:
    """Calculates total redundant forward pass ops across all segment recomputations."""
    if n_layers <= 0 or num_segments <= 0:
        return 0
    return n_layers * ops_per_layer
