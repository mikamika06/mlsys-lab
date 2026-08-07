def check(workdir):
    import ref
    from compcache.engine import CompilerEngine
    m = {"warmup_ok": 0.0}
    eng = CompilerEngine()
    oracle = ref.OracleEngine()
    graphs = [[5, 6], [7, 8]]
    eng.warmup(graphs)
    oracle.warmup(graphs)
    _, c1 = eng.compile_and_run(5)
    _, c2 = oracle.compile_and_run(5)
    if c1 == 0 and c2 == 0:
        m["warmup_ok"] = 1.0
    return m
