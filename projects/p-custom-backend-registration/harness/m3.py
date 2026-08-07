def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from compiler.backend import CompilationGraph
    from compiler.passes import CustomOptimizationPass
    m = {"pass_applied": 0.0}
    try:
        g = CompilationGraph([{"id": 1, "type": "target_op"}])
        p = CustomOptimizationPass()
        g_out = p.run(g)
        if g_out is not None and len(g_out.get_nodes()) == 1:
            m["pass_applied"] = 1.0
    except Exception:
        pass
    return m
