def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from compiler.backend import CompilationGraph
    m = {"graph_valid": 0.0}
    try:
        g = CompilationGraph([{"id": 1}])
        nodes = g.get_nodes()
        if isinstance(nodes, list) and len(nodes) == 1:
            m["graph_valid"] = 1.0
    except Exception:
        pass
    return m
