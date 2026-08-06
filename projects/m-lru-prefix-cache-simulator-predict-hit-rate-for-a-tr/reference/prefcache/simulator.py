from collections import OrderedDict

def simulate_lru_hit_rate(traces, capacity):
    cache = OrderedDict()
    hits = 0
    total = 0
    for trace in traces:
        for block in trace:
            total += 1
            if block in cache:
                cache.move_to_end(block)
                hits += 1
            else:
                if len(cache) >= capacity:
                    cache.popitem(last=False)
                cache[block] = True
    if total == 0:
        return 0.0
    return hits / total
