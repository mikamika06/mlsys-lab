def optimal_segment_count(n_layers: int) -> int:
    raise NotImplementedError


def analytical_peak_activation_memory(n_layers: int, num_segments: int, bytes_per_layer: int) -> int:
    raise NotImplementedError


def analytical_recompute_ops(n_layers: int, num_segments: int, ops_per_layer: int) -> int:
    raise NotImplementedError
