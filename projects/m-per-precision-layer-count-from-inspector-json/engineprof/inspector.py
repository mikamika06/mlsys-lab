def count_precisions(data):
    counts = {}
    for l in data.get("layers", []):
        p = l.get("precision", "UNKNOWN")
        counts[p] = counts.get(p, 0) + 1
    return counts
