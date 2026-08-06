def attribute_delta(off_run, on_run):
    dq = on_run["queue_ms"] - off_run["queue_ms"]
    dc = on_run["compute_ms"] - off_run["compute_ms"]
    dt = on_run["total_ms"] - off_run["total_ms"]
    return {"queue_delta": dq, "compute_delta": dc, "total_delta": dt}
