def check(workdir):
    import ref
    m = {"rewrite_ok": 0.0}
    eng = ref.get_sample_engine()
    eng.rewrite_op("custom_complex_op", lambda x: x)
    graph = ref.get_sample_graph()
    trace = eng.run(graph)
    share = eng.fallback_share(trace)
    if share < 0.9:
        m["rewrite_ok"] = 1.0
    return m
