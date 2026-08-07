def check(workdir):
    import ref
    from profiler.comparator import TraceComparator

    ta, tb, _ = ref.generate_traces()
    m = {"classification_ok": 0.0}
    try:
        comp = TraceComparator(ta, tb)
        cls = comp.classify_kernel("gemm_kernel")
        if cls in ["launch-bound", "compute-bound"]:
            m["classification_ok"] = 1.0
    except Exception:
        pass
    return m
