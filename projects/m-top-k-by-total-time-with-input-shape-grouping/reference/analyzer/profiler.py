import json


def load_trace(path):
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "traceEvents" in data:
        return data["traceEvents"]
    return data


def _to_tuple(val):
    if isinstance(val, list):
        return tuple(_to_tuple(x) for x in val)
    return val


def aggregate_by_shape(events):
    agg = {}
    for ev in events:
        if ev.get("ph") == "X" and "dur" in ev:
            name = ev.get("name", "")
            args = ev.get("args", {})
            dims = args.get("Input Dims", [])
            dims_t = _to_tuple(dims)
            key = (name, dims_t)
            if key not in agg:
                agg[key] = {"total_dur": 0, "count": 0}
            agg[key]["total_dur"] += ev["dur"]
            agg[key]["count"] += 1
    return agg


def top_k_by_total_time(events, k=5):
    agg = aggregate_by_shape(events)
    items = []
    for (name, shapes), metrics in agg.items():
        items.append({
            "name": name,
            "shapes": shapes,
            "total_dur": metrics["total_dur"],
            "count": metrics["count"]
        })
    items.sort(key=lambda x: (-x["total_dur"], x["name"], x["shapes"]))
    return items[:k]
