def check(workdir):
    import ref
    from compcache.engine import CompilerEngine
    m = {"zero_compilation_on_warm_request": 0.0}
    eng = CompilerEngine()
    graphs = [[42]]
    eng.warmup(graphs)
    _, comps = eng.compile_and_run(42)
    oracle = ref.OracleEngine()
    oracle.warmup(graphs)
    _, oracle_comps = oracle.compile_and_run(42)
    if comps == 0 and comps == oracle_comps:
        m["zero_compilation_on_warm_request"] = 1.0
    return m
