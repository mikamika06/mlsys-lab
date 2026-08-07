import ref


def check(workdir):
    from symshape.infer import propagate_shapes

    out = {"graphs_matched": 0.0}
    ok = 0
    total = len(ref.GRAPHS)

    for i, graph in enumerate(ref.GRAPHS):
        want = ref.propagate_shapes(graph)
        got = propagate_shapes(graph)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"graph {i}: got {got}, expected {want}"

    if ok == total:
        out["graphs_matched"] = 1.0

    return out
