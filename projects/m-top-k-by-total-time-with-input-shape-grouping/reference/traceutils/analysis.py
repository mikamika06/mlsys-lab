def group_and_aggregate(events):
    agg = {}
    for ev in events:
        key = (ev["name"], ev["input_shape"])
        agg[key] = agg.get(key, 0.0) + ev["total_time"]
    return agg


def top_k_by_shape(events, k=2):
    agg = group_and_aggregate(events)
    by_shape = {}
    for (name, shape), total in agg.items():
        by_shape.setdefault(shape, []).append((name, total))
    res = {}
    for shape, items in by_shape.items():
        items.sort(key=lambda x: x[1], reverse=True)
        res[shape] = items[:k]
    return res
