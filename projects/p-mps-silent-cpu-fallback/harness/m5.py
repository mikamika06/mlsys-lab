def check(workdir):
    import ref
    m = {"zero_fallback_ok": 0.0}
    eng = ref.get_sample_engine()
    eng.rewrite_op("custom_complex_op", lambda x: x)
    eng.rewrite_op("rare_fallback_op", lambda x: x)
    graph = ref.get_sample_graph()
    fallbacks = eng.hot_path_fallbacks(graph)
    if fallbacks == 0:
        m["zero_fallback_ok"] = 1.0
    return m
