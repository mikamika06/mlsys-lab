def check(workdir):
    import ref
    m = {"fallback_share_ok": 0.0}
    eng = ref.get_sample_engine()
    graph = ref.get_sample_graph()
    trace = eng.run(graph)
    share = eng.fallback_share(trace)
    if share > 0.9:
        m["fallback_share_ok"] = 1.0
    return m
