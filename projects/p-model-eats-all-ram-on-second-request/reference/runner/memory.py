class ModelMemoryProfiler:
    def __init__(self, weights_mb, num_layers, hidden_size, num_heads):
        self.weights_mb = weights_mb
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads

    def measure_footprint(self):
        overhead_mb = 256.0
        return self.weights_mb + overhead_mb

    def kv_cache_size_mb(self, num_ctx, slots):
        bytes_per_token_per_layer = 2 * (self.hidden_size * 2)
        total_bytes = num_ctx * slots * self.num_layers * bytes_per_token_per_layer
        return total_bytes / (1024 * 1024)


def analyze_duplicates(process_list):
    unique_pids = set(p["pid"] for p in process_list)
    if len(unique_pids) < len(process_list):
        return {"duplicate_detected": True, "reason": "fork_without_exec"}
    return {"duplicate_detected": False, "reason": "none"}


def optimize_config(budget_mb, weights_mb, num_layers, hidden_size, num_heads):
    overhead = 256.0
    available_for_kv = budget_mb - weights_mb - overhead
    if available_for_kv < 0:
        return {"num_ctx": 512, "slots": 1, "keep_alive": 300}

    bytes_per_token_per_layer = 2 * (hidden_size * 2)
    per_slot_bytes = 2048 * num_layers * bytes_per_token_per_layer
    per_slot_mb = per_slot_bytes / (1024 * 1024)

    slots = 2
    total_kv_needed = per_slot_mb * slots
    if total_kv_needed <= available_for_kv:
        return {"num_ctx": 2048, "slots": slots, "keep_alive": 300}
    return {"num_ctx": 1024, "slots": 1, "keep_alive": 300}
