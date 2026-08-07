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
