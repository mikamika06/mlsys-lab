def run_sweep(config):
    results = []
    for s in config.get("schemes", []):
        for w in config.get("workloads", []):
            results.append({"scheme": s["name"], "workload": w["id"], "metric": s.get("bits", 4) * w.get("intensity", 1.0)})
    return results
