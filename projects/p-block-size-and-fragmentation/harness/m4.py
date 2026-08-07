def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import blk.analysis as learner
    import ref

    m = {"trace_check_ok": 0.0}
    trace = [{"len": 25}, {"len": 64}, {"len": 12}]
    bs = 16
    try:
        got = learner.verify_trace_simulation(trace, bs)
        expected = ref.verify_trace_simulation(trace, bs)
        if got == expected:
            m["trace_check_ok"] = 1.0
    except Exception:
        pass
    return m
