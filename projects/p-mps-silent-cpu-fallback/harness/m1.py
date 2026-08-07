def check(workdir):
    import ref
    m = {"unimplemented_ok": 0.0}
    eng = ref.get_sample_engine()
    graph = ref.get_sample_graph()
    unimpl = eng.list_unimplemented_ops(graph)
    expected = ["custom_complex_op", "rare_fallback_op"]
    if sorted(unimpl) == sorted(expected):
        m["unimplemented_ok"] = 1.0
    return m
