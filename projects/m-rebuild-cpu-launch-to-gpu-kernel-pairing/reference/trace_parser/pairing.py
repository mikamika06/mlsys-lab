def pair_events(events):
    cpu_launches = {e.get("args", {}).get("correlation_id"): e for e in events if "correlation_id" in e.get("args", {}) and "Launch" in e.get("name", "")}
    gpu_kernels = {e.get("args", {}).get("correlation_id"): e for e in events if "correlation_id" in e.get("args", {}) and "kernel" in e.get("name", "")}
    pairs = []
    for cid, cpu in sorted(cpu_launches.items()):
        if cid in gpu_kernels:
            pairs.append((cpu["name"], gpu_kernels[cid]["name"]))
    return pairs
