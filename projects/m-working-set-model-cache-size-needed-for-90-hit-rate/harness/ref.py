from collections import Counter


def compute_working_set(trace, target_hit_rate=0.9):
    if not trace:
        return 0
    counts = Counter(trace)
    total = len(trace)
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    cumulative = 0
    needed = 0
    for item, freq in sorted_items:
        cumulative += freq
        needed += 1
        if cumulative / total >= target_hit_rate:
            break
    return needed


def warmup_curve(trace, cache_size):
    cache = []
    hits = []
    for item in trace:
        if item in cache:
            hits.append(1)
            cache.remove(item)
            cache.append(item)
        else:
            hits.append(0)
            if len(cache) >= cache_size:
                cache.pop(0)
            cache.append(item)
    return hits


CASES = [
    {"trace": [1, 2, 1, 3, 1, 2, 4, 1, 2, 1], "target": 0.9},
    {"trace": [10, 20, 10, 30, 10, 20, 10, 40, 10, 20, 10, 50], "target": 0.9},
    {"trace": [5] * 80 + [6] * 10 + [7] * 10, "target": 0.9},
]


CURVE_CASES = [
    {"trace": [1, 2, 1, 3, 1, 2, 1], "cache_size": 2},
    {"trace": [10, 11, 12, 10, 11, 13], "cache_size": 3},
    {"trace": [1, 2, 3, 1, 2, 3, 1], "cache_size": 2},
]
