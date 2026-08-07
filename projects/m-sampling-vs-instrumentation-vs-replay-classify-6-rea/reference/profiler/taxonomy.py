def classify_mechanisms(tools):
    return {name: cat for name, cat in tools}


def calculate_miss_bound(interval, duration, total):
    if duration >= interval:
        return 0.0
    return (interval - duration) / interval


def rank_profilers(profilers, metrics):
    return sorted(profilers, key=lambda p: metrics[p])
