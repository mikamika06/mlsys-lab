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

def reproduce_eviction_sequence(operations, capacity):
    cache = OrderedDict()
    evictions = []
    for op, block in operations:
        if op == "access":
            if block in cache:
                cache.move_to_end(block)
            else:
                if len(cache) >= capacity:
                    evicted_block, _ = cache.popitem(last=False)
                    evictions.append(evicted_block)
                cache[block] = True
        elif op == "free_reverse":
            if block in cache:
                del cache[block]
                evictions.append(block)
    return evictions

def compute_hit_rate_from_prometheus(counter_hits, counter_misses):
    total = counter_hits + counter_misses
    if total == 0:
        return 0.0
    return counter_hits / total

TRACES_SET = [
    ([[1, 2, 3, 1, 2, 3], [1, 4, 5, 1]], 3),
    ([[10, 20, 30, 40, 10, 20]], 2),
    ([[5, 6, 7, 8, 5, 9]], 4),
    ([[1, 2, 1, 2, 1, 2, 1]], 2),
    ([[100, 101, 102, 103, 104]], 3)
]

OPERATIONS_SET = [
    ([("access", 1), ("access", 2), ("access", 3), ("access", 4)], 2),
    ([("access", 10), ("access", 20), ("free_reverse", 10)], 1),
    ([("access", 5), ("access", 6), ("access", 7), ("free_reverse", 6), ("access", 8)], 2),
    ([("access", 1), ("access", 2), ("access", 1), ("free_reverse", 2)], 2),
    ([("access", 99), ("access", 100), ("access", 101), ("free_reverse", 100)], 2)
]
