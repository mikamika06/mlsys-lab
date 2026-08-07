import json
import random


def generate_trace(path, seed, num_events=1000):
    rng = random.Random(seed)
    events = []
    ops = [
        ("aten::mm", [[[32, 128], [128, 64]], [[64, 128], [128, 32]], [[16, 256], [256, 16]]]),
        ("aten::add", [[[32, 64], [32, 64]], [[128, 128], [128, 128]]]),
        ("aten::relu", [[[32, 64]], [[128, 128]]])
    ]
    ts = 0
    for _ in range(num_events):
        op, shapes = rng.choice(ops)
        shape = rng.choice(shapes)
        dur = rng.randint(10, 1000)
        if rng.random() < 0.1:
            events.append({"name": op, "ph": "B", "ts": ts})
        events.append({
            "name": op,
            "ph": "X",
            "ts": ts,
            "dur": dur,
            "args": {"Input Dims": shape}
        })
        ts += dur + rng.randint(1, 10)

    data = {"traceEvents": events}
    with open(path, "w") as f:
        json.dump(data, f)
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


def top_k_by_total_time(agg_dict, k=5):
    items = []
    for (name, shapes), metrics in agg_dict.items():
        items.append({
            "name": name,
            "shapes": shapes,
            "total_dur": metrics["total_dur"],
            "count": metrics["count"]
        })
    items.sort(key=lambda x: (-x["total_dur"], x["name"], x["shapes"]))
    return items[:k]
