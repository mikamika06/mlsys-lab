def peak_activation_memory(layers, base_mem, strategy, segment_size=None):
    raise NotImplementedError


def optimal_segment_size(layers, base_mem):
    raise NotImplementedError


def recompute_flops_overhead(layers, segment_size):
    raise NotImplementedError
