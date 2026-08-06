"""Cache hit ratio calculation from batch sequences."""

def compute_cache_hit_ratio(batch_sequence, cache_capacity=4):
    seen = []
    hits = 0
    misses = 0
    for b in batch_sequence:
        if b in seen:
            hits += 1
        else:
            misses += 1
            seen.append(b)
            if len(seen) > cache_capacity:
                seen.pop(0)
    total = hits + misses
    return hits / total if total > 0 else 0.0
