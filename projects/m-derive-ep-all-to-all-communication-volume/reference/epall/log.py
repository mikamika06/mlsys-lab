import numpy as np

def reconstruct_token_counts(log_records, world_size):
    counts = np.zeros((world_size, world_size), dtype=int)
    for rec in log_records:
        counts[rec["src"]][rec["dst"]] += rec["count"]
    return counts
