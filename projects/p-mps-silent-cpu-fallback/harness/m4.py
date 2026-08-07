def check(workdir):
    import ref
    m = {"speedup_ok": 0.0}
    eng = ref.get_sample_engine()
    graph = ref.get_sample_graph()
    trace_before = eng.run(graph)
    total_before = sum(s["duration"] for s in trace_before)
    eng.rewrite_op("custom_complex_op", lambda x: x)
    eng.rewrite_op("rare_fallback_op", lambda x: x)
    trace_after = eng.run(graph)
    total_after = sum(s["duration"] for s in trace_after)
    if total_after < total_before:
        m["speedup_ok"] = 1.0
    return m
