def cross_entropy_memory_share(trace):
    total_mem = 0
    ce_mem = 0
    for ev in trace.get("traceEvents", []):
        mem = ev.get("args", {}).get("memory", 0)
        total_mem += mem
        if "cross_entropy" in ev.get("name", ""):
            ce_mem += mem
    if total_mem == 0:
        return 0.0
    return float(ce_mem) / float(total_mem)
