def compute_hit_rate_from_prometheus(counter_hits, counter_misses):
    total = counter_hits + counter_misses
    if total == 0:
        return 0.0
    return counter_hits / total
