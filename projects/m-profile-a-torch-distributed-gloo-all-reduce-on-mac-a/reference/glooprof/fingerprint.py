def fingerprint_trace(trace_data):
    events = trace_data.get("events", [])
    for ev in events:
        args = ev.get("args", {})
        if "comm_pattern" in args:
            return args["comm_pattern"]
    return "dp"
