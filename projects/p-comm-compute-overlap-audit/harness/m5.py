import ref

def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from overlap import audit

    m = {"overlap_ratio_ok": 0.0}
    trace = [
        {"name": "comp", "start": 0, "dur": 100, "type": "compute"},
        {"name": "comm", "start": 20, "dur": 50, "type": "comm"}
    ]
    try:
        ratio = audit.compute_overlap_ratio(trace)
        if isinstance(ratio, float) and 0.0 <= ratio <= 1.0:
            m["overlap_ratio_ok"] = 1.0
    except Exception:
        pass
    return m
