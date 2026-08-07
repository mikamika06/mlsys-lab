def check(workdir):
    import ref
    from compcache.engine import CompilerEngine
    m = {"trace_match": 0.0}
    eng = CompilerEngine()
    oracle = ref.OracleEngine()
    graph = [1, 2, 3, 1, 2]
    res = eng.trace_ops(graph)
    exp = oracle.trace_ops(graph)
    if res == exp:
        m["trace_match"] = 1.0
    return m
