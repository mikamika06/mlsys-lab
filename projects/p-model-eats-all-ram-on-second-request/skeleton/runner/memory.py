class ModelMemoryProfiler:
    def __init__(self, weights_mb, num_layers, hidden_size, num_heads):
        raise NotImplementedError

    def measure_footprint(self):
        raise NotImplementedError

    def kv_cache_size_mb(self, num_ctx, slots):
        raise NotImplementedError


def analyze_duplicates(process_list):
    raise NotImplementedError


def optimize_config(budget_mb, weights_mb, num_layers, hidden_size, num_heads):
    raise NotImplementedError
