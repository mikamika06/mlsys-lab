def simulate_hit_rate(requests, pool_size):
    cache = []
    hits = 0
    for r in requests:
        if r in cache:
            hits += 1
            cache.remove(r)
            cache.append(r)
        else:
            if len(cache) >= pool_size:
                cache.pop(0)
            cache.append(r)
    return float(hits) / float(len(requests)) if requests else 0.0
