def check(workdir):
    import ref
    from profiler.comparator import TraceComparator

    ta, tb, _ = ref.generate_traces()
    m = {"sync_ok": 0.0}
    try:
        comp = TraceComparator(ta, tb)
        has_sync = comp.detect_synchronization()
        if isinstance(has_sync, bool):
            m["sync_ok"] = 1.0
    except Exception:
        pass
    return m
