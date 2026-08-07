def check(workdir):
    import ref
    from profiler.comparator import TraceComparator

    ta, tb, tc = ref.generate_traces()
    m = {"confirmation_ok": 0.0}
    try:
        comp = TraceComparator(ta, tb)
        confirmed = comp.confirm_root_cause(tc)
        if confirmed:
            m["confirmation_ok"] = 1.0
    except Exception:
        pass
    return m
