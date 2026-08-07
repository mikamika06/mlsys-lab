import numpy as np

def compute_warmup_curve(traces, cache_size):
    cache = set()
    hits = 0
    curve = []
    for i, t in enumerate(traces):
        if t in cache:
            hits += 1
        else:
            if len(cache) >= cache_size:
                cache.pop()
            cache.add(t)
        curve.append(hits / (i + 1))
    return curve
