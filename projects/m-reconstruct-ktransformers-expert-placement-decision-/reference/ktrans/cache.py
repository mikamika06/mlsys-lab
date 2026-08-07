def simulate_lru_cache(capacity, trace):
    if not trace or capacity <= 0:
        return 0.0
    cache = []
    hits = 0
    for item in trace:
        if item in cache:
            hits += 1
            cache.remove(item)
            cache.append(item)
        else:
            if len(cache) >= capacity:
                cache.pop(0)
            cache.append(item)
    return hits / len(trace)
