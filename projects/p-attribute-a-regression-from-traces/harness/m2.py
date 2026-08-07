def check(workdir):
    import ref
    from profiler.comparator import TraceComparator

    ta, tb, _ = ref.generate_traces()
    m = {"delta_ok": 0.0}
    try:
        comp = TraceComparator(ta, tb)
        kernel, delta = comp.find_max_delta()
        if kernel is not None and delta >= 0:
            m["delta_ok"] = 1.0
    except Exception:
        pass
    return m
