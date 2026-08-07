def select_eviction_policy(traces, capacity):
    policies = ["lru", "lfu", "longest-unused-subtree"]
    best_policy = "lru"
    best_hit_rate = -1.0
    for pol in policies:
        cache = []
        freq = {}
        last_used = {}
        hits = 0
        total = 0
        for step, token in enumerate(traces):
            total += 1
            freq[token] = freq.get(token, 0) + 1
            last_used[token] = step
            if token in cache:
                hits += 1
            else:
                if len(cache) >= capacity:
                    if pol == "lru":
                        evict_token = min(cache, key=lambda t: last_used.get(t, 0))
                    elif pol == "lfu":
                        evict_token = min(cache, key=lambda t: (freq.get(t, 0), last_used.get(t, 0)))
                    else:
                        evict_token = min(cache, key=lambda t: last_used.get(t, 0))
                    cache.remove(evict_token)
                cache.append(token)
        rate = hits / max(1, total)
        if rate > best_hit_rate:
            best_hit_rate = rate
            best_policy = pol
    return {"policy": best_policy, "hit_rate": best_hit_rate}
