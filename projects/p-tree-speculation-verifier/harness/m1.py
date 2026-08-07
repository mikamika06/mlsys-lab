def check(workdir):
    import speculation.tree as tree
    import ref

    m = {"merges_prefix": 0.0, "handles_branching": 0.0}

    cases = [
        [[1, 2], [1, 3]],
        [[1, 2, 4], [1, 2, 5], [1, 3], [8]]
    ]

    t1, p1 = tree.build_tree(cases[0])
    rt1, rp1 = ref.build_tree(cases[0])
    if list(t1) == list(rt1) and list(p1) == list(rp1):
        m["merges_prefix"] = 1.0

    t2, p2 = tree.build_tree(cases[1])
    rt2, rp2 = ref.build_tree(cases[1])
    if list(t2) == list(rt2) and list(p2) == list(rp2):
        m["handles_branching"] = 1.0

    return m
