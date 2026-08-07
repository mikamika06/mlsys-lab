def compute_host_to_device_latencies(x_events):
    host_map = {}
    device_map = {}
    for ev in x_events:
        cid = ev.get("args", {}).get("correlation_id")
        if cid is None:
            continue
        cat = ev.get("cat")
        if cat == "host_op":
            host_map[cid] = float(ev.get("ts", 0.0))
        elif cat == "gpu_op":
            device_map[cid] = float(ev.get("ts", 0.0))
    res = {}
    for cid in host_map:
        if cid in device_map:
            res[cid] = device_map[cid] - host_map[cid]
    return res
