def check(workdir):
    import ref
    m = {"shapes_ok": 0.0}
    model = {"a": ref.np.ones((10, 10))}
    p = ref.Pruner(model, [["a"]])
    p.prune_group(["a"], [0])
    if model["a"].shape == (9, 10):
        m["shapes_ok"] = 1.0
    return m
