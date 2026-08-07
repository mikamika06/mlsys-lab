def run_comparison(runners, workload):
    results = []
    for r in runners:
        name = r.get("name", "unknown")
        scale = r.get("scale", 1.0)
        tokens = [int(w * scale) for w in workload]
        results.append({"name": name, "tokens": tokens})
    return results
