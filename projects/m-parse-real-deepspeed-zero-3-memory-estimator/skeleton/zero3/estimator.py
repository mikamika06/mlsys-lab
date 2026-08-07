def estimate_zero3_memory(layer_specs, world_size, alignment_bytes=512, bytes_per_elem=2):
    raise NotImplementedError


def calculate_peak_forward_memory(layer_specs, world_size, prefetch_depth=1, bytes_per_elem=2):
    raise NotImplementedError
