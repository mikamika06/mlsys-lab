def compute_flops(events):
    flops_map = {}
    for e in events:
        if "flops" in e.get("args", {}):
            dur_us = e.get("dur", 1)
            dur_s = dur_us / 1000000.0
            flops_map[e["name"]] = float(e["args"]["flops"]) / dur_s
    return flops_map
