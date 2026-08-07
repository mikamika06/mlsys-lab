import ref

def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from overlap import audit

    m = {"barriers_found": 0.0}
    trace = [
        {"name": "comp", "start": 0, "dur": 100, "type": "compute"},
        {"name": "comm", "start": 150, "dur": 50, "type": "comm"}
    ]
    try:
        barriers = audit.find_barriers(trace)
        if isinstance(barriers, list) and len(barriers) > 0:
            m["barriers_found"] = float(len(barriers))
    except Exception:
        pass
    return m
