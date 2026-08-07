def compute_prefetch_schedule(layer_computes, layer_comm_times, prefetch_depth):
    raise NotImplementedError


def find_optimal_prefetch_depth(layer_computes, layer_comm_times, memory_per_layer, memory_limit):
    raise NotImplementedError
