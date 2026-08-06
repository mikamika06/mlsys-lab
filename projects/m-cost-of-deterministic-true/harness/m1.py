import ref

def check(workdir):
    from detcost.metrics import compute_latency_ratio, compute_memory_overhead
    out = {"latency_ratio": 0.0}
    ok = 0
    total = len(ref.CONFIGS)
    for cfg in ref.CONFIGS:
        want_lat = ref.get_latency_ratio(cfg)
        got_lat = compute_latency_ratio(cfg)
        got_mem = compute_memory_overhead(cfg)
        if abs(got_lat - want_lat) < 1e-5 and got_mem > 0:
            ok += 1
    if ok == total:
        out["latency_ratio"] = 1.0
    else:
        out["_note"] = f"matched {ok}/{total} configurations"
    return out
