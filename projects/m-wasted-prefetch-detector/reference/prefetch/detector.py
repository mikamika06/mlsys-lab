def detect_wasted(trace, l2_size=10):
    cache = []
    wasted = set()
    prefetched = set()
    for t, ev in enumerate(trace):
        bid = ev["block_id"]
        is_pref = ev["is_prefetch"]
        if is_pref:
            prefetched.add(bid)
            if bid not in cache:
                if len(cache) >= l2_size:
                    evicted = cache.pop(0)
                    if evicted in prefetched and evicted not in cache:
                        wasted.add(evicted)
                cache.append(bid)
        else:
            if bid in cache:
                cache.remove(bid)
                cache.append(bid)
            else:
                if len(cache) >= l2_size:
                    evicted = cache.pop(0)
                    if evicted in prefetched and evicted not in cache:
                        wasted.add(evicted)
                cache.append(bid)
    return sorted(list(wasted))
