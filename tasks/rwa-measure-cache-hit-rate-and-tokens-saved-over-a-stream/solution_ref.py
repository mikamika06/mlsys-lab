def measure_cache_stats(requests):
    seen = set()
    hits = 0
    reused_tokens = 0

    for req in requests:
        key = tuple(req)
        if key in seen:
            hits += 1
            reused_tokens += len(req)
        else:
            seen.add(key)

    hit_rate = hits / len(requests) if requests else 0.0
    return hit_rate, reused_tokens
