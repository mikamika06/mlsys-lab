def classify_slices(trace_data):
    events = trace_data.get("traceEvents", [])
    classes = {"compute": 0, "memory": 0, "overhead": 0}
    for ev in events:
        name = ev.get("name", "").lower()
        dur = ev.get("dur", 0)
        if "gemm" in name or "matmul" in name or "conv" in name:
            classes["compute"] += dur
        elif "memcpy" in name or "memset" in name:
            classes["memory"] += dur
        else:
            classes["overhead"] += dur
    return classes
