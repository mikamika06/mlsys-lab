def check(workdir):
    import ref
    from profiler.comparator import TraceComparator

    ta, tb, _ = ref.generate_traces()
    m = {"table_ok": 0.0}
    try:
        comp = TraceComparator(ta, tb)
        res = comp.reduce_trace(ta)
        if isinstance(res, dict) and "gemm_kernel" in res:
            m["table_ok"] = 1.0
    except Exception:
        pass
    return m
