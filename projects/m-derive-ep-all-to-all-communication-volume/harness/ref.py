import numpy as np

def compute_volume(world_size, num_tokens, hidden_size, top_k, dtype_size):
    total_elements = num_tokens * top_k * hidden_size
    per_rank_send = (total_elements * dtype_size) * (world_size - 1) / world_size
    return float(per_rank_send)

def reconstruct_counts(log_records, world_size):
    counts = np.zeros((world_size, world_size), dtype=int)
    for rec in log_records:
        counts[rec["src"]][rec["dst"]] += rec["count"]
    return counts

CONFIGS = [
    {"world_size": 8, "num_tokens": 2048, "hidden_size": 4096, "top_k": 2, "dtype_size": 2},
    {"world_size": 4, "num_tokens": 1024, "hidden_size": 2048, "top_k": 1, "dtype_size": 4},
    {"world_size": 16, "num_tokens": 4096, "hidden_size": 8192, "top_k": 2, "dtype_size": 2},
]

LOG_SAMPLES = [
    ([{"src": 0, "dst": 1, "count": 100}, {"src": 1, "dst": 0, "count": 150}], 2),
    ([{"src": 0, "dst": 2, "count": 50}, {"src": 2, "dst": 0, "count": 60}], 4),
]
