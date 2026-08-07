def check(workdir):
    import ref
    m = {"nodes_ok": 0.0, "edges_ok": 0.0}
    g = ref.DependencyGraph()
    g.add_node("a", (10, 10))
    g.add_edge("a", "b", 0, 0)
    if len(g.nodes) == 1:
        m["nodes_ok"] = 1.0
    if len(g.edges) == 1:
        m["edges_ok"] = 1.0
    return m
