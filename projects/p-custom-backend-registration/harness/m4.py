def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from compiler.backend import CompilationGraph
    from compiler.passes import check_equivalence
    m = {"equivalent": 0.0}
    try:
        g1 = CompilationGraph([{"id": 1}])
        g2 = CompilationGraph([{"id": 1}])
        if check_equivalence(g1, g2):
            m["equivalent"] = 1.0
    except Exception:
        pass
    return m
