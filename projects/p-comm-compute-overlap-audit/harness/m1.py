import ref

def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from overlap import audit

    m = {"events_extracted": 0.0, "comm_identified": 0.0}
    trace = ref.sample_trace()
    try:
        events = audit.extract_events(trace)
        if isinstance(events, list) and len(events) == len(trace):
            m["events_extracted"] = 1.0
        comm_events = [e for e in events if e.get("type") == "comm"]
        if len(comm_events) > 0:
            m["comm_identified"] = 1.0
    except Exception:
        pass
    return m
