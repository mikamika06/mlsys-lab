import random

def generate_trace(seed=42):
    rng = random.Random(seed)
    trace = []
    for t in range(500):
        block_id = rng.randint(0, 50)
        is_pref = rng.random() < 0.3
        trace.append({"time": t, "block_id": block_id, "is_prefetch": is_pref})
    return trace

def compute_reuse_histogram(trace, max_dist=100):
    last_seen = {}
    dists = [0] * (max_dist + 1)
    for t, ev in enumerate(trace):
        bid = ev["block_id"]
        if bid in last_seen:
            d = t - last_seen[bid]
            if d <= max_dist:
                dists[d] += 1
            else:
                dists[max_dist] += 1
        last_seen[bid] = t
    return dists

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

def model_ttft_savings(trace, l2_sizes):
    results = {}
    for size in l2_sizes:
        wasted = detect_wasted(trace, size)
        hits = 0
        total = 0
        cache = []
        for ev in trace:
            total += 1
            bid = ev["block_id"]
            if bid in cache:
                hits += 1
                cache.remove(bid)
                cache.append(bid)
            else:
                if len(cache) >= size:
                    cache.pop(0)
                cache.append(bid)
        saving = max(0.0, float(hits) / float(max(1, total)) * (1.0 - float(len(wasted)) / float(max(1, size))))
        results[size] = round(saving, 4)
    return results
