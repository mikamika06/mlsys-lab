import random

random.seed(42)

def _make_config():
    layers = []
    precisions = ["FP32", "FP16", "INT8"]
    for i in range(15):
        p = precisions[i % len(precisions)]
        layers.append({
            "index": i,
            "name": f"conv_{i}",
            "precision": p,
            "is_reformat": (i > 0 and precisions[(i-1) % len(precisions)] != p)
        })
    return {"layers": layers}

def _make_profile():
    records = []
    for i in range(10):
        records.append({
            "name": f"kernel_{i}",
            "time_ms": float((i + 1) * 5.5),
            "invocations": 2
        })
    return {"records": records}

CONFIGS = [_make_config() for _ in range(3)]
PROFILES = [_make_profile() for _ in range(3)]

def count_precisions(data):
    res = {}
    for l in data.get("layers", []):
        p = l.get("precision", "FP32")
        res[p] = res.get(p, 0) + 1
    return res

def aggregate_profile(data):
    total_time = 0.0
    total_invocations = 0
    for r in data.get("records", []):
        total_time += r.get("time_ms", 0.0) * r.get("invocations", 1)
        total_invocations += r.get("invocations", 1)
    return {
        "total_time": total_time,
        "total_invocations": total_invocations,
        "layer_count": len(data.get("records", []))
    }

def find_reformats(data):
    reformats = []
    layers = data.get("layers", [])
    for i in range(1, len(layers)):
        prev = layers[i-1].get("precision")
        curr = layers[i].get("precision")
        if prev != curr or layers[i].get("is_reformat", False):
            reformats.append(layers[i].get("index"))
    return sorted(list(set(reformats)))
