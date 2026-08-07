def check(workdir):
    import ref
    m = {"groups_ok": 0.0}
    g = ref.DependencyGraph()
    g.add_node("a", (10, 10))
    g.add_node("b", (10, 10))
    g.add_edge("a", "b", 0, 0)
    gf = ref.GroupFinder(g)
    groups = gf.find_groups()
    if len(groups) == 1 and len(groups[0]) == 2:
        m["groups_ok"] = 1.0
    return m
