def check(workdir):
    import ref
    m = {"runs_ok": 0.0}
    model = ref.get_reference_model()
    g = ref.DependencyGraph()
    g.add_node("layer1", (16, 16))
    g.add_node("layer2", (16, 16))
    g.add_edge("layer1", "layer2", 0, 0)
    gf = ref.GroupFinder(g)
    groups = gf.find_groups()
    p = ref.Pruner(model, groups)
    p.prune_group(groups[0], [0])
    if model["layer1"].shape[0] == 15:
        m["runs_ok"] = 1.0
    return m
