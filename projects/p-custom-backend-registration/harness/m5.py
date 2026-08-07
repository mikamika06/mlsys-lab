def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from compiler.backend import CompilationGraph
    from compiler.passes import CustomOptimizationPass
    m = {"target_nodes_hit": 0.0}
    try:
        g = CompilationGraph([{"id": 1, "type": "target_op"}])
        p = CustomOptimizationPass()
        g_out = p.run(g)
        nodes = g_out.get_nodes()
        if nodes[0].get("optimized") is True:
            m["target_nodes_hit"] = 1.0
    except Exception:
        pass
    return m
