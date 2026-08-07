def calculate_sharded_elements(layer_params, world_size):
    """
    Returns (fsdp_elements_per_gpu, zero3_elements_per_gpu)
    """
    raise NotImplementedError


def estimate_zero3_memory(layer_params, world_size):
    """
    Returns dict with keys: 'params', 'grads', 'os', 'total'
    representing bytes per GPU.
    """
    raise NotImplementedError


def calculate_prefetch_depth(layer_params, compute_times, bandwidth_bytes_per_sec):
    """
    Returns (gather_times_in_seconds, prefetch_depth_in_layers)
    """
    raise NotImplementedError
