from prefetch.detector import detect_wasted

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
