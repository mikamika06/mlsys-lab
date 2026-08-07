def check(workdir):
    import ref
    from compcache.engine import CompilerEngine
    m = {"cache_efficiency": 0.0}
    eng = CompilerEngine()
    oracle = ref.OracleEngine()
    graph = [10, 20, 10, 30, 20]
    eng_comps = 0
    for op in graph:
        _, c = eng.compile_and_run(op)
        eng_comps += c
    oracle_comps = 0
    for op in graph:
        _, c = oracle.compile_and_run(op)
        oracle_comps += c
    if eng_comps == oracle_comps and eng_comps < len(graph):
        m["cache_efficiency"] = 1.0
    return m
