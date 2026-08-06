import ref


def check(workdir):
    from graphguard.detector import detect_baked_ints
    graphs = ref.get_graphs()
    ok = 0
    for g in graphs:
        want = ref.detect_baked_ints(g)
        got = detect_baked_ints(g)
        if sorted(got) == sorted(want):
            ok += 1
    return {"exact_match": 1.0 if ok == len(graphs) else 0.0}
